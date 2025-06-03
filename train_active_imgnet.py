import argparse
import os
import random
import time
from datetime import timedelta

import clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.imagenet import ImageNet
from strategies import select_queries
from clap_imgnet import train_clap_teacher, evaluate_clap


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset name')
    parser.add_argument('--shots', type=int, default=1,
                        help='Number of shots for few-shot learning')
    parser.add_argument('--teacher_type', choices=['zs', 'clap'],
                        default='zs', help='teacher model type: zs (zero-shot), clap (Class Adaptive Linear Probing)')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='Path to teacher (Tip-Adapter) checkpoint')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--train_epoch', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='Root path for dataset')
    parser.add_argument('--log_dir', type=str, default='./logs/debug',
                        help='Log directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generator')

    # Active learning arguments
    parser.add_argument('--active_learning', action='store_true', default=True,
                        help='Enable active learning')
    parser.add_argument('--active_learning_rounds', type=int, default=3,
                        help='Number of active learning rounds')
    parser.add_argument('--active_learning_queries', type=int, default=1000,
                        help='Number of queries per active learning round')
    parser.add_argument('--active_learning_strategy', type=str, default='random',
                        choices=['random', 'coreset', 'uncertainty', 'classbalanced', 'pcoreset'],
                        help='Active learning strategy')
    parser.add_argument('--max_query_samples', type=int, default=3000,
                        help='Maximum number of samples to consider for query selection')

    # ! compatibility purpose
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')

    # Add CLAP teacher arguments
    parser.add_argument('--clap_constraint_weight', type=float, default=1.0,
                        help='Weight for CLAP constraint term (higher values preserve more zero-shot knowledge)')
    parser.add_argument('--clap_learning_rate', type=float, default=0.01,
                        help='Learning rate for CLAP teacher training')
    parser.add_argument('--clap_epochs', type=int, default=300,
                        help='Number of training epochs for CLAP teacher')
    parser.add_argument('--clap_lr_scheduler', type=str, default='cosine', choices=['cosine', 'none'],
                        help='Learning rate scheduler for CLAP teacher training')
    parser.add_argument('--clap_warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs for CLAP teacher training')
    parser.add_argument('--clap_alpha', type=float, default=0.5,
                        help='Alpha parameter for task residual in CLAP (0.0-1.0)')
    parser.add_argument('--clap_init_mode', type=str, default='clipweights', choices=['clipweights', 'zeros'],
                        help='How to initialize CLAP prototypes: with CLIP weights or zeros')
    parser.add_argument('--clap_distance', type=str, default='l2', choices=['l2', 'cosine'],
                        help='Distance metric for CLAP constraint term')
    parser.add_argument('--clap_constraint_mode', type=str, default='l2',
                        choices=['standard', 'balanced', 'corrected', 'constant', 'adaptative', 'l2'],
                        help='How to calculate class constraint weights in CLAP')

    args = parser.parse_args()


    # Print all arguments for debugging and logging purposes
    print("=== Arguments ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=================")

    return args

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        clip_weights_before_norm = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            # Prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings_before_norm = class_embeddings.clone()
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding_before_norm = class_embeddings_before_norm.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            clip_weights_before_norm.append(class_embedding_before_norm)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        clip_weights_before_norm = torch.stack(clip_weights_before_norm, dim=1).cuda()
    return clip_weights, clip_weights_before_norm


class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load DINO pre-trained ResNet50
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add two branches
        in_features = 2048
        self.ce_head = nn.Linear(in_features, num_classes)  # CE head
        self.kd_head = nn.Linear(in_features, num_classes)  # KD head

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Forward through both branches
        ce_out = self.ce_head(features)
        kd_out = self.kd_head(features)

        return ce_out, kd_out, features


def train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                  train_loader_labeled, train_loader_unlabeled, test_loader,
                  clip_weights, teacher_model=None):

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Setup training
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.train_epoch * len(train_loader_unlabeled))

    # Initialize training parameters
    temperature = 2.0
    best_acc = -100.0  # Track best accuracy for reporting only

    # Fixed alpha and beta values based on teacher type
    if args.teacher_type == 'clap':
        alpha = 0.2  # Alpha=0.2 for CLAP teacher
    else:
        alpha = 0.4  # Alpha=0.4 for other teachers
    beta = 0.5

    # Add strategy name to checkpoint path
    if args.active_learning:
        last_ckpt_path = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"ckpt_dho_{args.teacher_type}_{args.shots}shots_last.pt")
    else:
        last_ckpt_path = os.path.join(args.log_dir, args.dataset, str(args.seed), f"ckpt_dho_{args.teacher_type}_{args.shots}shots_last.pt")
    os.makedirs(os.path.dirname(last_ckpt_path), exist_ok=True)

    for epoch in range(args.train_epoch):
        student_model.train()
        total_loss = 0

        # Set samplers' epoch for proper shuffling
        if dist.is_initialized():
            train_loader_labeled.sampler.set_epoch(epoch)
            train_loader_unlabeled.sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\n📊 Training Epoch: {epoch+1}/{args.train_epoch}')

        labeled_iterator = iter(train_loader_labeled)

        for unlabeled_imgs, _ in tqdm(train_loader_unlabeled, desc=f'Training (lr={optimizer.param_groups[0]["lr"]:.6f})',
                                      disable=rank != 0, mininterval=0.0, miniters=10):
            # Get labeled batch (with cycling)
            try:
                labeled_imgs, labels = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(train_loader_labeled)
                labeled_imgs, labels = next(labeled_iterator)

            labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()

            # Normalize images
            labeled_imgs_student = student_normalize(labeled_imgs)
            unlabeled_imgs_student = student_normalize(unlabeled_imgs)
            labeled_imgs_clip = clip_normalize(labeled_imgs)
            unlabeled_imgs_clip = clip_normalize(unlabeled_imgs)

            # Get teacher predictions
            with torch.no_grad():
                if teacher_model is not None:
                    # Use Tip-Adapter teacher
                    clip_features = clip_model.encode_image(labeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_labeled = teacher_model(clip_features)

                    clip_features = clip_model.encode_image(unlabeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_unlabeled = teacher_model(clip_features)
                else:
                    # Use CLIP as teacher
                    clip_features = clip_model.encode_image(labeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_labeled = 100. * clip_features @ clip_weights

                    clip_features = clip_model.encode_image(unlabeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_unlabeled = 100. * clip_features @ clip_weights

            # Get student predictions
            stacked_imgs = torch.cat([labeled_imgs_student, unlabeled_imgs_student], dim=0)
            stacked_logits_ce, stacked_logits_kd, stacked_features = student_model(stacked_imgs)
            student_logits_labeled_ce = stacked_logits_ce[:labeled_imgs_student.size(0)]
            student_logits_labeled_kd = stacked_logits_kd[:labeled_imgs_student.size(0)]
            _ = stacked_logits_ce[labeled_imgs_student.size(0):]
            student_logits_unlabeled_kd = stacked_logits_kd[labeled_imgs_student.size(0):]

            # Calculate losses
            ce_loss = F.cross_entropy(student_logits_labeled_ce, labels)

            distill_loss_labeled = F.kl_div(
                F.log_softmax(student_logits_labeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_labeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            distill_loss_unlabeled = F.kl_div(
                F.log_softmax(student_logits_unlabeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_unlabeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # Combined loss with equal weights
            loss = 0.5 * ce_loss + 0.5 * (distill_loss_labeled + distill_loss_unlabeled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Gather total_loss from all processes
        if dist.is_initialized():
            total_loss = torch.tensor(total_loss, device='cuda')
            dist.all_reduce(total_loss)
            total_loss = total_loss.item() / dist.get_world_size()

        # Evaluate current epoch
        test_acc, _ = evaluate(student_model, test_loader, student_normalize, teacher_type=args.teacher_type)

        if rank == 0:
            avg_loss = total_loss / len(train_loader_unlabeled)
            print(f"Epoch: {epoch+1}/{args.train_epoch} | CE Loss: {ce_loss:.4f} | "
                  f"Distill Labeled: {distill_loss_labeled:.4f} | "
                  f"Distill Unlabeled: {distill_loss_unlabeled:.4f}")
            print(f"Total Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

            # Track best accuracy for reporting only
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

            # Save model at the end of each epoch (always override)
            torch.save({
                'model_state_dict': student_model.module.state_dict(),
                'epoch': epoch,
                'acc': test_acc,
                'alpha': alpha,
                'beta': beta
            }, last_ckpt_path)

    # Add a barrier to synchronize all processes before loading the last model
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("All processes synchronized before loading the last model.")

    # Load last checkpoint
    last_checkpoint = torch.load(last_ckpt_path)
    student_model.module.load_state_dict(last_checkpoint['model_state_dict'])
    final_acc, _ = evaluate(student_model, test_loader, student_normalize, teacher_type=args.teacher_type)

    # Final results
    if rank == 0:
        print("\n" + "=" * 60)
        print("🎯 FINAL RESULTS")
        print("=" * 60)
        print(f"📊 Best Test Accuracy During Training: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"⚙️  Fixed Parameters: α={alpha:.2f}, β={beta:.2f}")
        print(f"🎯 Final Test Accuracy: {final_acc:.2f}%")
        print("=" * 60 + "\n")

        # Add last checkpoint details
        print("=" * 60)
        print("🎯 LAST CHECKPOINT DETAILS")
        print("=" * 60)
        print(f"📊 Last Epoch: {last_checkpoint['epoch']+1}/{args.train_epoch}")
        print("=" * 60 + "\n")

    return final_acc, alpha, beta, final_acc


def evaluate(model, data_loader, normalize, teacher_type=None):
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Set fixed parameters based on teacher type
    if teacher_type == 'clap':
        alpha = 0.2  # Alpha=0.2 for CLAP teacher
    else:
        alpha = 0.4  # Alpha=0.4 for other teachers
    beta = 0.5   # Fixed beta value

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating',
                                   disable=dist.get_rank() if dist.is_initialized() else False):
            images, labels = images.cuda(), labels.cuda()
            images = normalize(images)
            outputs_ce, outputs_kd, features = model(images)

            # Convert logits to probabilities before interpolation
            probs_ce = F.softmax(outputs_ce, dim=1)
            probs_kd = F.softmax(outputs_kd / beta, dim=1)

            # Interpolate between CE and KD probabilities
            # Note: switching alpha calculation (alpha = 1 - alpha)
            probs = alpha * probs_ce + (1 - alpha) * probs_kd

            _, predicted = torch.max(probs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Gather metrics from all processes
    if dist.is_initialized():
        correct = torch.tensor(correct).cuda()
        total = torch.tensor(total).cuda()
        dist.all_reduce(correct)
        dist.all_reduce(total)
        correct = correct.item()
        total = total.item()

    acc = 100 * correct / total

    if rank == 0:
        print(f"Accuracy with fixed parameters (α={alpha:.2f}, β={beta:.2f}): {acc:.2f}%")

    return acc, acc  # Return the same accuracy twice for compatibility


def update_dataset(args, dataset, selected_indices, round):
    """
    Update the dataset by moving selected samples from unlabeled to labeled set.

    Args:
        args: Command line arguments
        dataset: Dataset object
        selected_indices: Indices of samples to move from unlabeled to labeled
        round: Current active learning round
    """
    # Get rank for distributed training
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Convert tensor to list if needed
    if torch.is_tensor(selected_indices):
        selected_indices = selected_indices.tolist()

    # Create directory structure: log_dir/dataset/seed/strategy/round_n
    round_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"round{round}")
    os.makedirs(round_dir, exist_ok=True)

    # Get current dataset sizes for reporting
    prev_train_x = len(dataset.train_x)
    prev_train_u = len(dataset.train_u)

    # Handle different types of dataset.train_u (regular dataset or Subset)
    if isinstance(dataset.train_u, torch.utils.data.Subset):
        # For Subset, we need to map the selected indices to the original dataset
        original_dataset = dataset.train_u.dataset
        indices_mapping = dataset.train_u.indices

        # Collect new labeled data
        new_labeled_data = []
        for idx in tqdm(selected_indices, desc="Collecting newly labeled data", disable=rank != 0):
            original_idx = indices_mapping[idx]
            new_labeled_data.append(original_dataset.imgs[original_idx])

        # Save the newly labeled indices to a file
        if rank == 0:
            with open(os.path.join(round_dir, 'new_labeled.txt'), 'w') as f:
                for idx in tqdm(selected_indices, desc="Saving newly labeled indices", disable=rank != 0):
                    original_idx = indices_mapping[idx]
                    img_path, label = original_dataset.imgs[original_idx]
                    f.write(f"{img_path} {label}\n")

        # Create a new subset excluding the selected indices
        selected_set = set(selected_indices)
        dataset.train_u = torch.utils.data.Subset(
            original_dataset,
            [indices_mapping[i] for i in range(len(dataset.train_u)) if i not in selected_set]
        )
    else:
        # For regular dataset, handle directly
        # Save the newly labeled indices to a file
        if rank == 0:
            with open(os.path.join(round_dir, 'new_labeled.txt'), 'w') as f:
                for idx in tqdm(selected_indices, desc="Saving newly labeled indices", disable=rank != 0):
                    img_path, label = dataset.train_u.imgs[idx]
                    f.write(f"{img_path} {label}\n")

        # Collect new labeled data
        new_labeled_data = []
        for idx in tqdm(selected_indices, desc="Collecting newly labeled data", disable=rank != 0):
            new_labeled_data.append(dataset.train_u.imgs[idx])

        # Create a mask for efficient filtering
        selected_set = set(selected_indices)
        dataset.train_u = torch.utils.data.Subset(
            dataset.train_u,
            [i for i in range(len(dataset.train_u)) if i not in selected_set]
        )

    # Save all labeled indices before update
    if rank == 0:
        with open(os.path.join(round_dir, 'before_labeled.txt'), 'w') as f:
            for img_path, label in tqdm(dataset.train_x.imgs, desc="Saving previously labeled indices", disable=rank != 0):
                f.write(f"{img_path} {label}\n")

    # Update the labeled dataset's internal lists
    dataset.train_x.imgs += new_labeled_data
    dataset.train_x.samples = dataset.train_x.imgs  # Update samples which is used by the dataset

    # Update targets if they exist
    if hasattr(dataset.train_x, 'targets'):
        new_targets = [label for _, label in new_labeled_data]
        dataset.train_x.targets += new_targets

    # Save all labeled indices after update
    if rank == 0:
        with open(os.path.join(round_dir, 'after_labeled.txt'), 'w') as f:
            for img_path, label in tqdm(dataset.train_x.imgs, desc="Saving all labeled indices", disable=rank != 0):
                f.write(f"{img_path} {label}\n")

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"🎯 DATASET UPDATE (Round {round})")
        print("=" * 60)
        print(f"📊 Labeled samples: {prev_train_x} -> {len(dataset.train_x)}")
        print(f"📊 Unlabeled samples: {prev_train_u} -> {len(dataset.train_u)}")
        print(f"📊 Added {len(selected_indices)} new samples to labeled set")
        print(f"📊 Saved indices to {round_dir}/[new|before|after]_labeled.txt")
        print("=" * 60 + "\n")


def active_learning_loop(args, student_model, student_normalize, clip_model, clip_normalize,
                         dataset, clip_weights, teacher_model=None):
    """
    Run the active learning loop.

    Args:
        args: Command line arguments
        student_model: Student model
        student_normalize: Normalization transform for student model
        clip_model: CLIP model
        clip_normalize: Normalization transform for CLIP model
        dataset: Dataset object
        clip_weights: CLIP weights
        teacher_model: Teacher model (optional)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Get clip_weights_before_norm for CLAP teacher
    _, clip_weights_before_norm = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Track metrics across rounds
    all_rounds_metrics = []

    # Create a reduced unlabeled set at the beginning
    if args.max_query_samples > 0 and args.max_query_samples < len(dataset.train_u):
        if rank == 0:
            print(f"Creating reduced unlabeled set of size {args.max_query_samples} from original size {len(dataset.train_u)}")

        # Use the same random seed across processes for consistent subset selection
        torch.manual_seed(args.seed)
        reduced_indices = torch.randperm(len(dataset.train_u))[:args.max_query_samples].tolist()

        # Create a subset of the unlabeled data
        dataset.train_u_full = dataset.train_u  # Store the full unlabeled set
        dataset.train_u = torch.utils.data.Subset(dataset.train_u, reduced_indices)

        if rank == 0:
            print(f"Reduced unlabeled set created with {len(dataset.train_u)} samples")

    for round in range(args.active_learning_rounds):
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"🔄 ACTIVE LEARNING ROUND {round+1}/{args.active_learning_rounds}")
            print("=" * 60)

        # Reinitialize the student model at the beginning of each round
        if rank == 0:
            print(f"🔄 Reinitializing student model for round {round+1}")

        # Create a new student model instance
        new_student_model = StudentModel(num_classes=len(dataset.classnames)).cuda()

        # Wrap with DDP if using distributed training
        if dist.is_initialized():
            new_student_model = DDP(new_student_model, device_ids=[args.local_rank])
            # Synchronize the new model across all processes
            dist.barrier()

        # Replace the old model with the new one
        student_model = new_student_model

        if rank == 0:
            print(f"✅ Student model reinitialized successfully")

        # Create a dataset that repeats the labeled dataset to match the unlabeled dataset size
        class RepeatedDataset(Dataset):
            def __init__(self, dataset, target_length):
                self.dataset = dataset
                self.target_length = target_length

            def __len__(self):
                return self.target_length

            def __getitem__(self, idx):
                return self.dataset[idx % len(self.dataset)]

        # Create repeated labeled dataset to match unlabeled dataset size
        repeated_labeled_dataset = RepeatedDataset(dataset.train_x, len(dataset.train_u))

        # Create samplers for distributed training
        if dist.is_initialized():
            train_sampler_labeled = DistributedSampler(repeated_labeled_dataset)
            train_sampler_unlabeled = DistributedSampler(dataset.train_u)
            val_sampler = DistributedSampler(dataset.val, shuffle=False)
            test_sampler = DistributedSampler(dataset.test, shuffle=False)
        else:
            train_sampler_labeled = None
            train_sampler_unlabeled = None
            val_sampler = None
            test_sampler = None

        # Create query subset for active learning
        if args.max_query_samples == -1:
            max_query_samples = len(dataset.train_u)
        else:
            # Use the entire reduced unlabeled set for query selection
            max_query_samples = len(dataset.train_u)

        if rank == 0:
            print(f"Using all {max_query_samples} samples from unlabeled pool for query selection")

        # Use all samples in the unlabeled pool for query selection
        query_indices = list(range(len(dataset.train_u)))
        query_subset = dataset.train_u  # Use the entire unlabeled set directly

        # Create sampler for the subset
        query_sampler = None  # Don't use DistributedSampler for query_loader since it's only used on rank 0
        support_sampler = None  # Don't use DistributedSampler for support_loader since it's only used on rank 0

        # Use the repeated dataset instead of dataset.train_x
        train_loader_labeled = torch.utils.data.DataLoader(
            repeated_labeled_dataset, batch_size=args.batch_size,
            sampler=train_sampler_labeled,
            shuffle=train_sampler_labeled is None,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        train_loader_unlabeled = torch.utils.data.DataLoader(
            dataset.train_u, batch_size=args.batch_size,
            sampler=train_sampler_unlabeled,
            shuffle=train_sampler_unlabeled is None,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        val_loader = torch.utils.data.DataLoader(
            dataset.val, batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        # Add support and query loaders with no augmentation for active learning
        support_loader = torch.utils.data.DataLoader(
            dataset.train_x, batch_size=args.batch_size,
            sampler=support_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )
        query_loader = torch.utils.data.DataLoader(
            query_subset, batch_size=args.batch_size,
            sampler=query_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        # Create a standard non-repeated loader for the labeled data for teacher training
        train_loader_labeled_std = torch.utils.data.DataLoader(
            dataset.train_x, batch_size=args.batch_size,
            sampler=DistributedSampler(dataset.train_x) if dist.is_initialized() else None,
            shuffle=not dist.is_initialized(),
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        # Retrain the teacher model if using CLAP
        if args.teacher_type == 'clap':
            if rank == 0:
                print("\n" + "=" * 60)
                print(f"🔄 TRAINING CLAP TEACHER FOR ROUND {round+1}")
                print("=" * 60)

            # Only train teacher model on main process in DDP mode
            if args.local_rank <= 0:  # Main process or single GPU mode
                # Train CLAP teacher model on a single GPU (cuda:0)
                clap_dict = train_clap_teacher(
                    args, clip_model, train_loader_labeled_std, val_loader, test_loader,
                    clip_weights, clip_weights_before_norm, clip_normalize, round=round
                )

                # Create a function to get teacher predictions
                def get_teacher_predictions(features):
                    with torch.no_grad():
                        prototypes = clap_dict['prototypes'].to(features.device)
                        # Always normalize prototypes
                        prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                        features_norm = features / features.norm(dim=-1, keepdim=True)
                        logit_scale = clip_model.logit_scale.exp()
                        return features_norm @ prototypes_norm * logit_scale

                # Save teacher model to file for other processes to load
                if args.local_rank != -1:
                    teacher_save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy)
                    teacher_save_path = os.path.join(teacher_save_dir, f"clap_teacher_last_{args.shots}shots_round{round}.pt")
                    os.makedirs(teacher_save_dir, exist_ok=True)
                    torch.save({
                        'prototypes': clap_dict['prototypes'],
                        'val_acc': clap_dict['val_acc'],
                        'test_acc': clap_dict['test_acc'],
                    }, teacher_save_path)

                teacher_model = get_teacher_predictions

                if rank == 0:
                    print(f"✅ CLAP teacher model trained for round {round+1} with test accuracy {clap_dict['test_acc']:.2f}%")
            elif args.local_rank > 0:  # Secondary processes in DDP
                # Wait for the main process to finish training and save the model
                teacher_save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy)
                teacher_save_path = os.path.join(teacher_save_dir, f"clap_teacher_last_{args.shots}shots_round{round}.pt")

                # Wait for the file to be created by the main process
                while not os.path.exists(teacher_save_path):
                    print(f"Process {args.local_rank}: Waiting for teacher model to be saved...")
                    time.sleep(10)  # Wait 10 seconds before checking again

                print(f"Process {args.local_rank}: Loading teacher model from {teacher_save_path}")
                clap_dict = torch.load(teacher_save_path, map_location=f"cuda:{args.local_rank}")

                # Create a function to get teacher predictions
                def get_teacher_predictions(features):
                    with torch.no_grad():
                        prototypes = clap_dict['prototypes'].to(features.device)
                        # Always normalize prototypes
                        prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                        features_norm = features / features.norm(dim=-1, keepdim=True)
                        logit_scale = clip_model.logit_scale.exp()
                        return features_norm @ prototypes_norm * logit_scale

                teacher_model = get_teacher_predictions
                print(f"Process {args.local_rank}: Loaded teacher model with validation accuracy {clap_dict['val_acc']:.2f}%")

            # Synchronize all processes before continuing
            if args.local_rank != -1:
                dist.barrier()

        # Use more epochs for the final round
        original_train_epoch = args.train_epoch
        if round == args.active_learning_rounds - 1:
            args.train_epoch = 50
            if rank == 0:
                print(f"🔄 Final round: Using {args.train_epoch} epochs for training")

        # Train student model
        final_acc, alpha, beta, _ = train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                      train_loader_labeled, train_loader_unlabeled, test_loader,
                      clip_weights, teacher_model=teacher_model)

        # Store metrics for this round
        round_metrics = {
            'round': round + 1,
            'labeled_samples': len(dataset.train_x),
            'unlabeled_samples': len(dataset.train_u),
            'test_acc': final_acc,
            'alpha': alpha,
            'beta': beta,
        }
        all_rounds_metrics.append(round_metrics)

        # Save round checkpoint
        if rank == 0:
            # Create directory for round checkpoints
            round_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"round{round}")
            os.makedirs(round_dir, exist_ok=True)

            # Save model checkpoint for this round
            round_ckpt_path = os.path.join(round_dir, f"student_model_round{round+1}.pt")
            ckpt_path = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"ckpt_dho_{args.teacher_type}_{args.shots}shots_round{round+1}.pt")
            if os.path.exists(ckpt_path):
                last_checkpoint = torch.load(ckpt_path)
                torch.save({
                    'round': round + 1,
                    'model_state_dict': student_model.module.state_dict(),
                    'acc': final_acc,
                    'alpha': alpha,
                    'beta': beta,
                    'labeled_samples': len(dataset.train_x),
                    'unlabeled_samples': len(dataset.train_u),
                    'metrics': round_metrics
                }, round_ckpt_path)
                print(f"✅ Saved round {round+1} checkpoint to {round_ckpt_path}")
            else:
                print(f"❌ Round {round+1} checkpoint not found at {ckpt_path}")

            # Also save the last model checkpoint for this round
            last_round_ckpt_path = os.path.join(round_dir, f"student_model_round{round+1}_last.pt")
            last_ckpt_path = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"ckpt_dho_{args.teacher_type}_{args.shots}shots_last.pt")
            if os.path.exists(last_ckpt_path):
                last_checkpoint = torch.load(last_ckpt_path)
                torch.save({
                    'round': round + 1,
                    'model_state_dict': last_checkpoint['model_state_dict'],
                    'acc': last_checkpoint['acc'],
                    'alpha': last_checkpoint['alpha'],
                    'beta': last_checkpoint['beta'],
                    'labeled_samples': len(dataset.train_x),
                    'unlabeled_samples': len(dataset.train_u)
                }, last_round_ckpt_path)
                print(f"✅ Saved last checkpoint for round {round+1} to {last_round_ckpt_path}")
            else:
                print(f"❌ Last checkpoint not found at {last_ckpt_path}")

        # Restore original epoch count
        args.train_epoch = original_train_epoch

        # Skip query selection in the last round
        if round < args.active_learning_rounds - 1:
            # Collect predictions on unlabeled data
            query_pool = []

            # Only collect predictions on the main process
            if rank == 0:
                student_model.eval()

                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(tqdm(query_loader,
                                                                      desc="Collecting predictions on rank 0")):
                        images = images.cuda()
                        images_student = student_normalize(images)

                        # Get student predictions
                        outputs_ce, outputs_kd, features = student_model(images_student)

                        # Get features for feature-based methods
                        features = student_model.module.backbone(images_student)
                        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = features.view(features.size(0), -1)

                        # Get teacher logits for images if available
                        teacher_logits = None
                        with torch.no_grad():
                            # Apply normalization for CLIP model
                            images_clip = clip_normalize(images)
                            clip_features = clip_model.encode_image(images_clip)
                            clip_features /= clip_features.norm(dim=-1, keepdim=True)
                            if teacher_model is not None:
                                # Use custom teacher model
                                teacher_logits = teacher_model(clip_features).cpu()
                            else:
                                # Use CLIP as teacher (zero-shot)
                                teacher_logits = (100. * clip_features @ clip_weights).cpu()

                        # Store batch information with global indices
                        global_indices = torch.tensor([query_indices[batch_idx * args.batch_size + i]
                                                     for i in range(min(args.batch_size,
                                                                       len(query_subset) - batch_idx * args.batch_size))])

                        batch_info = {
                            'indices': global_indices,
                            'ce_out': outputs_ce.cpu(),
                            'kd_out': outputs_kd.cpu(),
                            'features': features.cpu(),
                            'images': images.cpu(),
                            'teacher_logits': teacher_logits,  # Add teacher logits for KD-based methods
                            'labels': targets.cpu()  # Add ground truth labels for class-based strategies
                        }
                        query_pool.append(batch_info)

                print(f"Query pool size on rank 0: {len(query_pool)}")

            # Wait for rank 0 to finish collecting predictions
            if dist.is_initialized():
                dist.barrier()

            # Select queries only on rank 0
            if rank == 0:
                selected_indices = select_queries(
                    args,
                    query_pool,
                    strategy=args.active_learning_strategy,
                    support_loader=support_loader,
                    student_normalize=student_normalize,
                    student_model=student_model
                )

                # Create directory structure: log_dir/dataset/seed/strategy/round_n
                round_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"round{round}")
                os.makedirs(round_dir, exist_ok=True)

                # Save selected indices to file for other processes to read
                indices_file = os.path.join(round_dir, f"round{round+1}_selected_indices.pt")
                torch.save(selected_indices, indices_file)
            else:
                selected_indices = None
                round_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"round{round}")

            # Broadcast selected indices to all processes
            if dist.is_initialized():
                # Wait for rank 0 to finish selecting indices
                dist.barrier()

                # Non-root processes load the selected indices
                if rank != 0:
                    indices_file = os.path.join(round_dir, f"round{round+1}_selected_indices.pt")
                    selected_indices = torch.load(indices_file)

                # Make sure all processes have loaded the indices
                dist.barrier()

            # All processes update their datasets simultaneously
            update_dataset(args, dataset, selected_indices, round+1)

            # Load the last checkpoint for query selection
            last_ckpt_path = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"ckpt_dho_{args.teacher_type}_{args.shots}shots_last.pt")
            if os.path.exists(last_ckpt_path) and rank == 0:
                print(f"Loading last checkpoint for query selection: {last_ckpt_path}")
                last_checkpoint = torch.load(last_ckpt_path)
                student_model.module.load_state_dict(last_checkpoint['model_state_dict'])
                print(f"Last checkpoint loaded (epoch {last_checkpoint['epoch']})")

            # Synchronize model loading across processes
            if dist.is_initialized():
                dist.barrier()

    # Print final statistics
    if rank == 0:
        print("\n" + "=" * 80)
        print("🎯 ACTIVE LEARNING FINAL RESULTS")
        print("=" * 80)
        print(f"Strategy: {args.active_learning_strategy}")
        print(f"Total rounds: {args.active_learning_rounds}")
        print(f"Queries per round: {args.active_learning_queries}")
        alpha_value = 0.2 if args.teacher_type == 'clap' else 0.4
        print(f"Fixed parameters: α={alpha_value}, β=0.5 (α is weight of CE branch, 1-α is weight of KD branch)")
        print("=" * 80)
        print("📊 PERFORMANCE ACROSS ROUNDS")
        print("-" * 80)
        print(f"{'Round':<10}{'Labeled Samples':<20}{'Test Accuracy':<20}")
        print("-" * 80)

        for metrics in all_rounds_metrics:
            print(f"{metrics['round']:<10}{metrics['labeled_samples']:<20}{metrics['test_acc']:.2f}%")

        print("-" * 80)

        # Calculate improvement
        if len(all_rounds_metrics) > 1:
            initial_acc = all_rounds_metrics[0]['test_acc']
            final_acc = all_rounds_metrics[-1]['test_acc']
            improvement = final_acc - initial_acc
            print(f"Initial accuracy: {initial_acc:.2f}%")
            print(f"Final accuracy: {final_acc:.2f}%")
            print(f"Improvement: {improvement:.2f}% ({improvement/initial_acc*100:.2f}% relative)")

        print("=" * 80 + "\n")

def main():
    # Load arguments
    args = get_arguments()
    process_count = int(os.environ.get('WORLD_SIZE', 1))
    args.batch_size = args.batch_size // process_count
    args.local_rank = int(os.environ.get('RANK', -1))
    print(f'Using distributed training with {process_count} processes. Batch size per process: {args.batch_size}')

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800000))

    # Setup transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Setup transforms
    student_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # Prepare ImageNet dataset
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(args.root_path, args.shots, train_transform, val_transform)

    # Load CLIP model
    clip_model, _ = clip.load('RN50')
    clip_model = clip_model.to(f"cuda:{args.local_rank}")

    # Get CLIP weights
    clip_weights, clip_weights_before_norm = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    clip_weights = clip_weights.to(f"cuda:{args.local_rank}")
    clip_weights_before_norm = clip_weights_before_norm.to(f"cuda:{args.local_rank}")
    clip_model.eval()

    # Initialize DINO student model with multi-GPU support
    student_model = StudentModel(num_classes=len(imagenet.classnames)).cuda()
    student_model = DDP(student_model, device_ids=[args.local_rank])

    # Load teacher model based on type
    teacher_model = None
    if args.teacher_type == 'clap':
        print("Using CLAP teacher model")
        # No need to initialize CLAP here since it will be trained in each active learning round
        teacher_model = None  # Will be initialized in active_learning_loop

    elif args.teacher_ckpt:
        print(f"Loading teacher model from {args.teacher_ckpt}")
        teacher_ckpt = torch.load(args.teacher_ckpt, map_location=f"cuda:{args.local_rank}")
        device = torch.device(f"cuda:{args.local_rank}")

        adapter_weight = teacher_ckpt['adapter_weight'].to(device)
        best_beta = teacher_ckpt['best_beta']
        best_alpha = teacher_ckpt['best_alpha']
        _ = teacher_ckpt['cache_keys'].to(device)
        cache_values = teacher_ckpt['cache_values'].to(device)
        clip_weights = teacher_ckpt['clip_weights'].to(device)

        # Create a function to get teacher predictions
        def get_teacher_predictions(features, adapter_weight=adapter_weight, cache_values=cache_values,
                                    clip_weights=clip_weights, best_beta=best_beta, best_alpha=best_alpha):
            with torch.no_grad():
                affinity = features @ adapter_weight.t()
                cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                return clip_logits + cache_logits * best_alpha

        teacher_model = get_teacher_predictions

        print(f"Teacher model loaded successfully from {args.teacher_ckpt}")
    else:
        # Use CLIP as teacher for zs
        print("Using CLIP as teacher (zero-shot)")
        clip_weights = clip_weights.to(f"cuda:{args.local_rank}")
        clip_model.eval()

    if args.active_learning:
        # Run active learning loop
        active_learning_loop(args, student_model, student_normalize, clip_model, clip_normalize,
                            imagenet, clip_weights, teacher_model=teacher_model)
    else:
        # Create samplers for distributed training
        train_sampler_labeled = DistributedSampler(imagenet.train_x) if dist.is_initialized() else None
        val_sampler = DistributedSampler(imagenet.val, shuffle=False) if dist.is_initialized() else None
        test_sampler = DistributedSampler(imagenet.test, shuffle=False) if dist.is_initialized() else None
        train_sampler_unlabeled = DistributedSampler(imagenet.train_u) if dist.is_initialized() else None

        # Create dataloaders
        train_loader_labeled = torch.utils.data.DataLoader(
            imagenet.train_x, batch_size=args.batch_size,
            sampler=train_sampler_labeled,
            shuffle=train_sampler_labeled is None,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        val_loader = torch.utils.data.DataLoader(
            imagenet.val, batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        test_loader = torch.utils.data.DataLoader(
            imagenet.test, batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        train_loader_unlabeled = torch.utils.data.DataLoader(
            imagenet.train_u, batch_size=args.batch_size,
            sampler=train_sampler_unlabeled,
            shuffle=train_sampler_unlabeled is None,
            num_workers=8, pin_memory=True,
            persistent_workers=True,
            worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
        )

        # If using CLAP, train it first
        if args.teacher_type == 'clap':
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print("\n" + "=" * 60)
                print("🔄 TRAINING CLAP TEACHER")
                print("=" * 60)

            # Only train teacher model on main process in DDP mode
            if args.local_rank <= 0:  # Main process or single GPU mode
                clap_dict = train_clap_teacher(
                    args, clip_model, train_loader_labeled, val_loader, test_loader,
                    clip_weights, clip_weights_before_norm, clip_normalize
                )

                # Create a function to get teacher predictions
                def get_teacher_predictions(features):
                    with torch.no_grad():
                        prototypes = clap_dict['prototypes'].to(features.device)
                        prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                        features_norm = features / features.norm(dim=-1, keepdim=True)
                        logit_scale = clip_model.logit_scale.exp()
                        return features_norm @ prototypes_norm * logit_scale

                # Save teacher model to file for other processes to load
                if args.local_rank != -1:
                    teacher_save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed))
                    teacher_save_path = os.path.join(teacher_save_dir, f"clap_teacher_last_{args.shots}shots.pt")
                    os.makedirs(teacher_save_dir, exist_ok=True)
                    torch.save({
                        'prototypes': clap_dict['prototypes'],
                        'val_acc': clap_dict['val_acc'],
                        'test_acc': clap_dict['test_acc'],
                    }, teacher_save_path)

                teacher_model = get_teacher_predictions

                if rank == 0:
                    print(f"✅ CLAP teacher model trained with test accuracy {clap_dict['test_acc']:.2f}%")
            elif args.local_rank > 0:  # Secondary processes in DDP
                # Wait for the main process to finish training and save the model
                teacher_save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy)
                teacher_save_path = os.path.join(teacher_save_dir, f"clap_teacher_last_{args.shots}shots_round{round}.pt")

                # Wait for the file to be created by the main process
                while not os.path.exists(teacher_save_path):
                    print(f"Process {args.local_rank}: Waiting for teacher model to be saved...")
                    time.sleep(10)  # Wait 10 seconds before checking again

                print(f"Process {args.local_rank}: Loading teacher model from {teacher_save_path}")
                clap_dict = torch.load(teacher_save_path, map_location=f"cuda:{args.local_rank}")

                # Create a function to get teacher predictions
                def get_teacher_predictions(features):
                    with torch.no_grad():
                        prototypes = clap_dict['prototypes'].to(features.device)
                        prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                        features_norm = features / features.norm(dim=-1, keepdim=True)
                        logit_scale = clip_model.logit_scale.exp()
                        return features_norm @ prototypes_norm * logit_scale

                teacher_model = get_teacher_predictions
                print(f"Process {args.local_rank}: Loaded teacher model with validation accuracy {clap_dict['val_acc']:.2f}%")

            # Synchronize all processes before continuing
            if args.local_rank != -1:
                dist.barrier()

        # Train student model
        train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                    train_loader_labeled, train_loader_unlabeled, test_loader,
                    clip_weights, teacher_model=teacher_model)


if __name__ == '__main__':
    main()

