import argparse
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import transformers
from tqdm import tqdm

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from strategies import select_queries
from clap_others import train_clap_teacher, evaluate_clap


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='caltech101',
                        choices=['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101'],
                        help='Dataset name')
    parser.add_argument('--shots', type=int, default=1,
                        help='Number of shots (samples per class) for few-shot learning')

    parser.add_argument('--teacher_type', choices=['zs', 'clap'],
                        default='zs', help='teacher model type: zs (zero-shot), clap (Class Adaptive Linear Probing)')
    parser.add_argument('--teacher_ckpt', type=str, default="",
                        help='Path to CLAP teacher checkpoint')
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

    parser.add_argument('--student_model', choices=['res18', 'mobilenet', 'tiny_vit'], default='res18',
                        help='Student model architecture (ResNet18, MobileNetV2, or Tiny ViT)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--train_epoch', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='Root path for dataset')
    parser.add_argument('--log_dir', type=str, default='./logs/debug',
                        help='Directory to save logs')
    parser.add_argument('--save_memory', action='store_true',
                        help='Enable memory-saving mode for active learning')

    parser.add_argument('--active_learning_rounds', type=int, default=3,
                        help='Number of active learning rounds')
    parser.add_argument('--active_learning_strategy', type=str, default='pcoreset',
                        choices=['random', 'coreset', 'uncertainty', 'badge', 'classbalanced', 'pcoreset'],
                        help='Strategy for selecting queries in active learning')
    parser.add_argument('--active_learning_queries', type=int, default=100,
                        help='Number of queries per active learning round')

    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')

    parser.add_argument('--delete_checkpoint_after_use', action='store_true',
                      help='Delete model checkpoint files after evaluation to save disk space')

    args = parser.parse_args()

    # Print all arguments
    print("\nArguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    return args


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        # clip_weights = []
        clip_weights_before_norm = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            # Prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings_before_norm = class_embeddings.clone()
            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            class_embedding_before_norm = class_embeddings_before_norm.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            # clip_weights.append(class_embedding)
            clip_weights_before_norm.append(class_embedding_before_norm)

        clip_weights_before_norm = torch.stack(clip_weights_before_norm, dim=1).cuda()
        # clip_weights = torch.stack(clip_weights, dim=1).cuda()
        clip_weights = clip_weights_before_norm / clip_weights_before_norm.norm(dim=0, keepdim=True)
    return clip_weights, clip_weights_before_norm


class StudentModel(nn.Module):
    def __init__(self, num_classes, model_type='res18'):
        super().__init__()
        # Choose backbone based on model type
        if model_type == 'res18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = 512
            self.is_vit = False
        elif model_type == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=True)
            in_features = 1280
            self.is_vit = False
        elif model_type == 'tiny_vit':
            # Load Tiny ViT model from timm
            self.backbone = timm.create_model(
                'tiny_vit_21m_384.dist_in22k_ft_in1k',
                pretrained=True
            )
            in_features = self.backbone.head.in_features
            # Remove the classification head
            self.backbone.head.fc = nn.Identity()
            self.is_vit = True

        if self.is_vit == False:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add two branches
        self.ce_head = nn.Linear(in_features, num_classes)
        self.kd_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Get features from backbone
        if self.is_vit:
            # For Tiny ViT, the backbone already has the head removed
            features = self.backbone(x)
        else:
            # For CNN models, use the existing approach
            features = self.backbone(x)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Forward through both branches
        ce_out = self.ce_head(features)
        kd_out = self.kd_head(features)

        return ce_out, kd_out, features


def train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                  train_loader_labeled, train_loader_unlabeled, val_loader,
                  test_loader, clip_weights, teacher_model=None, round=0):
    # Setup training
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)

    # Adjust the number of epochs and create a loader that repeats data 5x per epoch
    effective_epochs = args.train_epoch // 200
    # Make sure we have at least one epoch
    effective_epochs = max(1, effective_epochs)

    # Instead of manually repeating batches in memory, use a more efficient approach
    # Get the dataset from the loader
    labeled_dataset = train_loader_labeled.dataset

    # Calculate repetition factor
    repeat_factor = 200

    # Create a data loader that samples with replacement
    # This achieves the same effect as repeating the dataset many times
    train_loader_labeled_repeated = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=train_loader_labeled.batch_size,
        sampler=torch.utils.data.RandomSampler(
            labeled_dataset,
            replacement=True,
            num_samples=len(labeled_dataset) * repeat_factor
        ),
        num_workers=train_loader_labeled.num_workers if hasattr(train_loader_labeled, 'num_workers') else 4,
        pin_memory=True
    )

    print(f"Created efficient repeated data loader with repeat factor: {repeat_factor}")
    print(f"Total training steps: {len(train_loader_labeled_repeated) * effective_epochs}")

    # Update scheduler to account for the new effective number of steps
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * effective_epochs * len(train_loader_labeled_repeated),
        num_training_steps=effective_epochs * len(train_loader_labeled_repeated)
    )

    # Initialize training parameters
    temperature = 2.0
    # Fixed alpha and beta values
    alpha, beta = 0.5, 1.0
    # Update the cache path to follow the consistent pattern
    save_dir = os.path.join(args.log_dir, args.active_learning_strategy, str(round))
    os.makedirs(save_dir, exist_ok=True)
    # Add path for last checkpoint
    last_ckpt_path = os.path.join(save_dir, f"last_ckpt_actkd_{args.student_model}_{args.teacher_type}_{args.shots}shots_seed{args.seed}.pt")

    training_history = {
        'epochs': [],
        'train_losses': [],
        'val_accuracies': [],
        'best_val_acc': 0.0  # Just for tracking, not for model selection
    }

    # Main training loop - now using effective epochs
    for epoch in range(effective_epochs):
        student_model.train()
        total_loss = 0
        total_ce_loss = 0
        total_distill_labeled_loss = 0
        total_distill_unlabeled_loss = 0

        print(f'\n📊 Training Epoch: {epoch+1}/{effective_epochs} (effective epoch)')

        # Create an iterator for the unlabeled data loader
        unlabeled_iter = iter(train_loader_unlabeled)

        # Process the repeated labeled data with corresponding unlabeled data
        for i, (labeled_data) in enumerate(tqdm(
            train_loader_labeled_repeated,
            total=len(train_loader_labeled_repeated),
            desc='Training batches'
        )):
            # Get unlabeled batch, cycling through the unlabeled loader if needed
            try:
                unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(train_loader_unlabeled)
                unlabeled_data = next(unlabeled_iter)

            # Unpack data
            labeled_imgs, labels = labeled_data
            unlabeled_imgs, _ = unlabeled_data

            # Move data to GPU
            labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()

            # Prepare inputs for both student and teacher models
            labeled_imgs_student = student_normalize(labeled_imgs)
            unlabeled_imgs_student = student_normalize(unlabeled_imgs)
            labeled_imgs_clip = clip_normalize(labeled_imgs)
            unlabeled_imgs_clip = clip_normalize(unlabeled_imgs)

            # Generate teacher predictions
            with torch.no_grad():
                # Extract and normalize CLIP features
                labeled_feats = clip_model.encode_image(labeled_imgs_clip)
                labeled_feats = labeled_feats / labeled_feats.norm(dim=-1, keepdim=True)
                unlabeled_feats = clip_model.encode_image(unlabeled_imgs_clip)
                unlabeled_feats = unlabeled_feats / unlabeled_feats.norm(dim=-1, keepdim=True)

                # Generate teacher logits (either from teacher model or CLIP)
                teacher_logits_labeled = teacher_model(labeled_feats) if teacher_model else 100. * labeled_feats @ clip_weights
                teacher_logits_unlabeled = teacher_model(unlabeled_feats) if teacher_model else 100. * unlabeled_feats @ clip_weights

            # Generate student predictions from both heads
            student_logits_labeled_ce, student_logits_labeled_kd, _ = student_model(labeled_imgs_student)
            _, student_logits_unlabeled_kd, _ = student_model(unlabeled_imgs_student)

            # Calculate training losses
            # 1. Standard cross-entropy loss on labeled data
            ce_loss = F.cross_entropy(student_logits_labeled_ce, labels)

            # 2. Knowledge distillation loss on labeled data
            distill_loss_labeled = F.kl_div(
                F.log_softmax(student_logits_labeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_labeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # 3. Knowledge distillation loss on unlabeled data
            distill_loss_unlabeled = F.kl_div(
                F.log_softmax(student_logits_unlabeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_unlabeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # Combine losses with equal weights
            loss = 0.5 * ce_loss + 0.5 * (distill_loss_labeled + distill_loss_unlabeled)

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_distill_labeled_loss += distill_loss_labeled.item()
            total_distill_unlabeled_loss += distill_loss_unlabeled.item()

        # Calculate average loss for this epoch
        avg_loss = total_loss / len(train_loader_labeled_repeated)
        avg_ce_loss = total_ce_loss / len(train_loader_labeled_repeated)
        avg_distill_labeled_loss = total_distill_labeled_loss / len(train_loader_labeled_repeated)
        avg_distill_unlabeled_loss = total_distill_unlabeled_loss / len(train_loader_labeled_repeated)

        # Store basic metrics in history
        training_history['epochs'].append(epoch + 1)
        training_history['train_losses'].append(avg_loss)

        # Print basic training info every epoch
        print(f"Epoch: {epoch+1}/{effective_epochs} | CE Loss: {avg_ce_loss:.4f} | Distill Labeled: {avg_distill_labeled_loss:.4f} | Distill Unlabeled: {avg_distill_unlabeled_loss:.4f}")

        # Evaluate on validation set every 5 epochs or on the last epoch, but only for monitoring
        if (epoch + 1) % 5 == 0 or epoch == effective_epochs - 1:
            val_acc = evaluate(student_model, val_loader, student_normalize)
            print(f"Total Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% (α: 0.5, β: 1.0)")

            # Store validation metrics for monitoring only
            training_history['val_accuracies'].append(val_acc)
            if val_acc > training_history['best_val_acc']:
                training_history['best_val_acc'] = val_acc

    # Save the last model checkpoint
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'alpha': alpha,
        'beta': beta
    }, last_ckpt_path)

    # Use the last model for evaluation
    last_checkpoint = torch.load(last_ckpt_path, weights_only=False)
    student_model.load_state_dict(last_checkpoint['model_state_dict'])

    # Evaluate test accuracy with fixed alpha=0.5 and beta=1.0
    test_acc = evaluate(student_model, test_loader, student_normalize)

    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS (Round {round})")
    print("=" * 60)
    print(f"📊 Labeled samples: {len(train_loader_labeled.dataset)}")
    print(f"📊 Unlabeled samples: {len(train_loader_unlabeled.dataset)}")
    print("=" * 60)
    print(f"📊 Best Validation Accuracy (monitoring only): {training_history['best_val_acc']:.2f}%")
    print(f"⚙️  Fixed Parameters: α=0.5, β=1.0")
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")
    print("=" * 60 + "\n")

    training_history['final_test_acc'] = test_acc
    training_history['alpha'] = 0.5
    training_history['beta'] = 1.0

    # Delete model checkpoint after evaluation to save disk space
    if os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
            print(f"✅ Removed checkpoint file {last_ckpt_path} to save disk space")
        except Exception as e:
            print(f"❌ Failed to remove checkpoint file: {e}")

    return training_history

def evaluate(model, data_loader, normalize, alpha=0.5, beta=1.0):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            images = normalize(images)
            outputs_ce, outputs_kd, _ = model(images)

            # Convert logits to probabilities before interpolation
            probs_ce = F.softmax(outputs_ce, dim=1)
            probs_kd = F.softmax(outputs_kd / beta, dim=1)

            # Interpolate between CE and KD probabilities with fixed alpha=0.5, beta=1.0
            probs = (1 - alpha) * probs_ce + alpha * probs_kd

            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_query_pool(args, student_model, student_normalize, clip_model,
                     teacher_normalize, unlabeled_loader, clip_weights, teacher_model=None):
    """Creates a pool of potential queries from unlabeled data"""
    student_model.eval()
    query_pool = []
    start_idx = 0

    print(f"Building query pool with memory-saving mode: {args.save_memory}")

    with torch.no_grad():
        for images, labels in tqdm(unlabeled_loader, desc="Building query pool"):
            images = images.cuda()
            batch_size = images.size(0)

            # Create indices for this batch
            indices = torch.arange(start_idx, start_idx + batch_size)
            start_idx += batch_size

            # Prepare inputs for both student and teacher models
            images_student = student_normalize(images)
            images_clip = teacher_normalize(images)

            ce_out, kd_out, features = student_model(images_student)

            # Get teacher predictions
            clip_feats = clip_model.encode_image(images_clip)
            clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
            teacher_logits = teacher_model(clip_feats) if teacher_model else 100. * clip_feats @ clip_weights

            # Store data - don't store images if save_memory is enabled
            if args.save_memory:
                query_pool.append({
                    'indices': indices,
                    'features': features.cpu(),
                    'ce_out': ce_out.cpu(),
                    'kd_out': kd_out.cpu(),
                    'teacher_logits': teacher_logits.cpu(),
                    'labels': labels.cpu()
                })
            else:
                query_pool.append({
                    'images': images.cpu(),
                    'indices': indices,
                    'features': features.cpu(),
                    'ce_out': ce_out.cpu(),
                    'kd_out': kd_out.cpu(),
                    'teacher_logits': teacher_logits.cpu(),
                    'labels': labels.cpu()
                })
    return query_pool

def update_dataset(args, dataset, selected_indices, round):
    """Updates the dataset by moving selected samples from unlabeled to labeled set"""
    # Move selected samples from train_u to train_x
    new_labeled_data = []
    remaining_unlabeled = []

    for idx, item in enumerate(dataset.train_u):
        if idx in selected_indices:
            new_labeled_data.append(item)
        else:
            remaining_unlabeled.append(item)

    prev_train_x = len(dataset.train_x)
    prev_train_u = len(dataset.train_u)

    # Update dataset
    dataset.train_x.extend(new_labeled_data)
    dataset._train_u = remaining_unlabeled

    print("\n" + "=" * 60)
    print(f"🎯 DATASET UPDATE (Round {round})")
    print("=" * 60)
    print(f"📊 Labeled samples: {prev_train_x} -> {len(dataset.train_x)}")
    print(f"📊 Unlabeled samples: {prev_train_u} -> {len(dataset.train_u)}")
    print("=" * 60 + "\n")




def main():
    # Load arguments
    args = get_arguments()

    # Setup cache directory - update to follow the consistent pattern
    # We'll create the base directory here, and specific round directories will be created as needed
    os.makedirs(args.log_dir, exist_ok=True)

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    student_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    teacher_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))


    # Load dataset
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)

    # Load CLIP model and teacher model
    clip_model, _ = clip.load('RN50')
    clip_model.eval()

    # Get CLIP weights
    clip_weights, clip_weights_before_norm = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Determine which teacher to use
    teacher_model = None

    # Use CLAP teacher
    if args.teacher_type == 'clap':
        print("Using CLAP teacher model that will be trained and updated each round")
        # First train a CLAP teacher model using the available labeled data
        clap_dict = train_clap_teacher(
            args, clip_model, dataset,
            clip_weights_before_norm, teacher_normalize, round=0
        )

        # Create the teacher model function
        def get_teacher_predictions(features):
            with torch.no_grad():
                prototypes = clap_dict['prototypes']
                # Always normalize prototypes
                prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                features_norm = features / features.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                return features_norm @ prototypes_norm * logit_scale

        # Set the teacher model
        teacher_model = get_teacher_predictions

    # Load specified CLAP teacher checkpoint if provided
    elif args.teacher_ckpt:
        print(f"Loading CLAP teacher model from {args.teacher_ckpt}")
        clap_ckpt = torch.load(args.teacher_ckpt)

        prototypes = clap_ckpt['prototypes']
        clip_weights = clap_ckpt['clip_weights']

        # Create the teacher model function
        def get_teacher_predictions(features):
            with torch.no_grad():
                # Always normalize prototypes
                prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                features_norm = features / features.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                return features_norm @ prototypes_norm * logit_scale

        teacher_model = get_teacher_predictions

        print(f"CLAP teacher model loaded successfully from {args.teacher_ckpt}")

    # Use zero-shot teacher (CLIP model directly)
    else:
        print("Using zero-shot teacher (CLIP model directly)")
        # Create a function to get teacher predictions using zero-shot CLIP
        def get_teacher_predictions(features):
            with torch.no_grad():
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                # Get zero-shot predictions
                return 100. * features @ clip_weights

        teacher_model = get_teacher_predictions

    # Add CLIP-related arguments to args
    args.clip_model = clip_model
    args.teacher_model = teacher_model
    args.teacher_normalize = teacher_normalize
    args.clip_weights = clip_weights
    args.clip_weights_before_norm = clip_weights_before_norm

    # Extract labels from the labeled dataset
    labels = [item.label for item in dataset.train_x]

    # Count occurrences of each class
    num_classes = len(dataset.classnames)
    class_counts = torch.zeros(num_classes, dtype=torch.int)
    for label in labels:
        class_counts[label] += 1

    # Create data loaders
    train_loader_labeled = build_data_loader(
        dataset.train_x, batch_size=args.batch_size,
        tfm=train_transform, is_train=True, shuffle=True, drop_last=True, duplicate_if_needed=True
    )
    train_loader_unlabeled = build_data_loader(
        dataset.train_u, batch_size=args.batch_size,
        tfm=train_transform, is_train=True, shuffle=True, drop_last=True, duplicate_if_needed=True
    )
    val_loader = build_data_loader(
        dataset.val, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )
    test_loader = build_data_loader(
        dataset.test, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )

    # If saving memory, use a smaller batch size for the query loader
    query_batch_size = min(args.batch_size, 32) if args.save_memory else args.batch_size
    print(f"Using query batch size: {query_batch_size} (memory-saving mode: {args.save_memory})")

    query_loader = build_data_loader(
        dataset.train_u, batch_size=query_batch_size,
        tfm=val_transform, is_train=False, shuffle=False
    )
    support_loader = build_data_loader(
        dataset.train_x, batch_size=args.batch_size,
        tfm=val_transform, is_train=False, shuffle=False
    )

    # Train student model
    student_model = StudentModel(num_classes=len(dataset.classnames), model_type=args.student_model).cuda()
    all_results = []

    # Create directory for initial training (round 0)
    save_dir = os.path.join(args.log_dir, args.active_learning_strategy, "0")
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate CLIP zero-shot accuracy on test set
    print("\n" + "=" * 60)
    print("📊 EVALUATING CLIP ZERO-SHOT ACCURACY")
    print("=" * 60)

    def evaluate_clip_zeroshot(model, weights, loader, normalize):
        model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating CLIP zero-shot"):
                images, labels = images.cuda(), labels.cuda()
                images = normalize(images)

                # Get image features
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Calculate logits using the CLIP weights
                logits = 100. * image_features @ weights

                # Get predictions
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        return 100 * correct / total

    # Run zero-shot evaluation
    clip_zs_acc = evaluate_clip_zeroshot(clip_model, clip_weights, test_loader, teacher_normalize)
    print(f"CLIP Zero-Shot accuracy on test set: {clip_zs_acc:.2f}%")

    # Initial training (round 0)
    print("\n" + "=" * 60)
    print("🚀 INITIAL TRAINING (ROUND 0)")
    print("=" * 60)

    # Train initial model
    history = train_student(args, student_model, student_normalize, clip_model,
                          teacher_normalize, train_loader_labeled,
                          train_loader_unlabeled, val_loader, test_loader,
                          clip_weights, teacher_model=teacher_model, round=0)
    all_results.append(history)

    # Active learning loop
    for round in range(args.active_learning_rounds):
        print(f"\n🔄 Active Learning Round {round+1}/{args.active_learning_rounds}")

        # Create query pool from unlabeled data
        query_pool = get_query_pool(args, student_model, student_normalize, clip_model, teacher_normalize, query_loader, clip_weights)

        # Select queries with updated arguments
        selected_indices = select_queries(args, query_pool, args.active_learning_strategy,
                                       support_loader, student_normalize, student_model)

        # Update dataset with selected queries
        update_dataset(args, dataset, selected_indices, round)

        # Create new data loaders with updated dataset
        train_loader_labeled = build_data_loader(
            dataset.train_x, batch_size=args.batch_size,
            tfm=train_transform, is_train=True, shuffle=True, drop_last=True, duplicate_if_needed=True
        )
        train_loader_unlabeled = build_data_loader(
            dataset.train_u, batch_size=args.batch_size,
            tfm=train_transform, is_train=True, shuffle=True, drop_last=True, duplicate_if_needed=True
        )
        query_loader = build_data_loader(
            dataset.train_u, batch_size=query_batch_size,
            tfm=val_transform, is_train=False, shuffle=False
        )
        support_loader = build_data_loader(
            dataset.train_x, batch_size=args.batch_size,
            tfm=val_transform, is_train=False, shuffle=False
        )

        # Train model on updated dataset with reinitialization
        print(f"📚 Training with {len(dataset.train_x)} labeled samples")
        student_model = StudentModel(num_classes=len(dataset.classnames), model_type=args.student_model).cuda()
        history = train_student(args, student_model, student_normalize, clip_model,
                              teacher_normalize, train_loader_labeled,
                              train_loader_unlabeled, val_loader, test_loader,
                              clip_weights, teacher_model=teacher_model, round=round+1)
        all_results.append(history)

        # Train a new CLAP teacher model if using CLAP
        if args.teacher_type == 'clap':
            print(f"\n🔄 Training new CLAP teacher model for Round {round+1}")
            clap_dict = train_clap_teacher(
                args, clip_model, dataset,
                clip_weights_before_norm, teacher_normalize, round=round+1
            )

            # Update the teacher model function
            def get_teacher_predictions(features):
                with torch.no_grad():
                    prototypes = clap_dict['prototypes']
                    # Always normalize prototypes
                    prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
                    features_norm = features / features.norm(dim=-1, keepdim=True)
                    logit_scale = clip_model.logit_scale.exp()
                    return features_norm @ prototypes_norm * logit_scale

            # Update the teacher model
            teacher_model = get_teacher_predictions
            args.teacher_model = teacher_model
            print(f"✅ Updated CLAP teacher model with {len(dataset.train_x)} labeled samples")

    # Print final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TRAINING SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Initial shots per class: {args.shots}")
    print(f"Teacher model type: {args.teacher_type}")

    # Print teacher-specific info
    if args.teacher_type == 'clap':
        print(f"CLAP constraint weight: {args.clap_constraint_weight}")
        print(f"CLAP initialization mode: {args.clap_init_mode}")
        # Task residual is removed
        print(f"CLAP distance metric: {args.clap_distance}")
        print(f"CLAP constraint mode: {args.clap_constraint_mode}")
        print(f"CLAP learning rate: {args.clap_learning_rate}")
        print(f"CLAP epochs: {args.clap_epochs}")
        print(f"CLAP LR scheduler: {args.clap_lr_scheduler}")
        if args.clap_lr_scheduler == 'cosine':
            print(f"CLAP warmup epochs: {args.clap_warmup_epochs}")
    else:
        print(f"Using zero-shot CLIP teacher")

    print(f"Active learning strategy: {args.active_learning_strategy}")
    print(f"Queries per round: {args.active_learning_queries}")
    print("-" * 80)
    print("Results per round:")
    for round, history in enumerate(all_results):
        print(f"Round {round}: Best val acc: {history.get('best_val_acc', 0):.2f}%, Test acc: {history.get('final_test_acc', 0):.2f}%")

    # Make sure to remove any remaining model checkpoints
    cleanup_paths = []
    for root, dirs, files in os.walk(args.log_dir):
        for file in files:
            if file.endswith(".pt") and ("ckpt_actkd" in file or "last_ckpt" in file):
                cleanup_paths.append(os.path.join(root, file))

    if cleanup_paths:
        print(f"\n🧹 Found {len(cleanup_paths)} checkpoint files to clean up")
        for path in cleanup_paths:
            try:
                os.remove(path)
                print(f"✅ Removed: {path}")
            except Exception as e:
                print(f"❌ Failed to remove {path}: {e}")

    # Save final summary to a file in the main strategy directory
    summary_path = os.path.join(args.log_dir, args.active_learning_strategy, "final_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Initial shots per class: {args.shots}\n")
        f.write(f"Teacher model type: {args.teacher_type}\n")

        # Write teacher-specific info
        if args.teacher_type == 'clap':
            f.write(f"CLAP constraint weight: {args.clap_constraint_weight}\n")
            f.write(f"CLAP initialization mode: {args.clap_init_mode}\n")
            # Task residual is removed
            f.write(f"CLAP distance metric: {args.clap_distance}\n")
            f.write(f"CLAP constraint mode: {args.clap_constraint_mode}\n")
            f.write(f"CLAP learning rate: {args.clap_learning_rate}\n")
            f.write(f"CLAP epochs: {args.clap_epochs}\n")
            f.write(f"CLAP LR scheduler: {args.clap_lr_scheduler}\n")
            if args.clap_lr_scheduler == 'cosine':
                f.write(f"CLAP warmup epochs: {args.clap_warmup_epochs}\n")
        else:
            f.write(f"Using zero-shot CLIP teacher\n")

        f.write(f"Active learning strategy: {args.active_learning_strategy}\n")
        f.write(f"Queries per round: {args.active_learning_queries}\n")
        f.write("-" * 80 + "\n")
        f.write("Results per round:\n")
        for round, history in enumerate(all_results):
            f.write(f"Round {round}: Best val acc: {history.get('best_val_acc', 0):.2f}%, Test acc: {history.get('final_test_acc', 0):.2f}%\n")

    print(f"Final summary saved to {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
