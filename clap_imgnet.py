import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from tqdm import tqdm


def train_clap_teacher(args, clip_model, train_loader_labeled, val_loader, test_loader, clip_weights, clip_weights_before_norm, teacher_normalize, round=0):
    """
    Train a CLAP (CLass Adaptive Linear Probing) teacher model using the labeled data.
    This implements the constraint-based approach from the CLAP paper.

    Current settings:
    - Initialization mode: 'clipweights' (initialize with CLIP weights)
    - Distance metric: 'l2' (L2 distance)
    - Constraint mode: 'l2' (L2 constraint)
    - Constraint weight: 1.0 (weight for the constraint term)

    These settings balance learning from labeled data while preserving zero-shot knowledge
    for classes where CLIP already performs well.

    Args:
        args: Arguments
        clip_model: CLIP model
        train_loader_labeled: DataLoader for labeled data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        clip_weights: CLIP text weights
        clip_weights_before_norm: CLIP text weights before normalization
        teacher_normalize: Normalization transform for teacher (CLIP) model
        round: Current active learning round

    Returns:
        Dict with the trained CLAP model parameters
    """
    # Check if we're in distributed training
    is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
    rank = dist.get_rank() if is_distributed else 0

    # Only run CLAP teacher training on main process (rank 0)
    if rank != 0:
        print(f"Process {rank}: Skipping CLAP teacher training (will wait for main process)")
        if is_distributed:
            # Return a dummy result that will be overwritten when results are loaded from disk
            return {'prototypes': None, 'val_acc': 0.0, 'test_acc': 0.0}

    print("\n" + "=" * 60)
    print(f"🚀 TRAINING CLAP TEACHER (ROUND {round})")
    print("=" * 60)
    print(f"📚 Training with {len(train_loader_labeled.dataset)} labeled samples")

    # Create directory for teacher models
    if args.active_learning:
        save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), args.active_learning_strategy, f"round{round}", "teacher")
    else:
        save_dir = os.path.join(args.log_dir, args.dataset, str(args.seed), "teacher")
    os.makedirs(save_dir, exist_ok=True)
    last_ckpt_path = os.path.join(save_dir, f"clap_teacher_last_{args.shots}shots_round{round}.pt")

    # Always use cuda:0 for teacher training (no DDP)
    device = torch.device("cuda:0")

    # Move clip model to the teacher GPU
    clip_model = clip_model.to(device)

    # Create non-DDP dataloaders for teacher model training
    # Extract the dataset from the DDP dataloaders
    train_dataset = train_loader_labeled.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    # Create new dataloaders without DDP
    teacher_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    teacher_val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    teacher_test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    # Initialize model parameters with CLIP weights before normalization
    print("Initializing CLAP prototypes with CLIP weights before normalization")
    clip_weights = clip_weights.to(device)
    clip_weights_before_norm = clip_weights_before_norm.to(device)
    prototypes = clip_weights_before_norm.clone().detach()
    prototypes.requires_grad = True

    # Hyperparameters for CLAP
    alpha_constraint = torch.zeros(clip_weights.shape[1]).to(device)

    # Precompute validation features for efficient evaluation (only for logging purposes)
    print("Precomputing validation features for efficient evaluation (for logging only)...")
    val_features = []
    val_targets = []
    with torch.no_grad():
        for images, targets in tqdm(teacher_val_loader, desc="Precomputing validation features", mininterval=0.0, miniters=10):
            images = images.to(device)
            targets = targets.to(device)
            # Apply teacher normalization
            images = teacher_normalize(images)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            val_features.append(features)
            val_targets.append(targets)

    # Concatenate all batches
    val_features = torch.cat(val_features, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    print(f"Precomputed features for {val_features.size(0)} validation samples")

    # Precompute test features for efficient evaluation (only for logging purposes)
    print("Precomputing test features for efficient evaluation (for logging only)...")
    test_features = []
    test_targets = []
    with torch.no_grad():
        for images, targets in tqdm(teacher_test_loader, desc="Precomputing test features", mininterval=0.0, miniters=10):
            images = images.to(device)
            targets = targets.to(device)
            # Apply teacher normalization
            images = teacher_normalize(images)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            test_features.append(features)
            test_targets.append(targets)

    # Concatenate all batches
    test_features = torch.cat(test_features, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    print(f"Precomputed features for {test_features.size(0)} test samples")

    # Rest of the function remains the same
    # Evaluate zero-shot CLIP accuracy
    with torch.no_grad():
        # Compute logits
        zs_logits = 100. * test_features @ clip_weights
        # Compute accuracy
        zeroshot_acc = (zs_logits.argmax(dim=1) == test_targets).float().mean().item() * 100
        print(f"Zero-shot Accuracy: {zeroshot_acc:.2f}%")

    # Create a non-augmented dataloader for constraint calculation
    train_loader_no_aug = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    # Extract features from non-augmented training set
    print("Extracting features from non-augmented training set for constraint initialization...")
    train_features_no_aug, train_labels_no_aug = [], []
    with torch.no_grad():
        for images, targets in tqdm(train_loader_no_aug, desc="Computing class-adaptive constraints", mininterval=0.0, miniters=10):
            images = images.to(device)
            targets = targets.to(device)

            # Apply teacher normalization
            images = teacher_normalize(images)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

            train_features_no_aug.append(features)
            train_labels_no_aug.append(targets)

        train_features_no_aug = torch.cat(train_features_no_aug)
        train_labels_no_aug = torch.cat(train_labels_no_aug)

        # Get zero-shot logits for training set
        train_zs_logits = 100. * train_features_no_aug @ clip_weights
        train_zs_logits = train_zs_logits.float()

    # Initialize lagrangian multipliers
    print("Getting initial lagrangian multipliers for constraint formulation")
    with torch.no_grad():
        # Get one-hot encoding ground-truth
        train_labels_one_hot = F.one_hot(train_labels_no_aug, num_classes=clip_weights.shape[1]).float()

        # Get zero_shot performance
        performance = torch.diag(torch.softmax(train_zs_logits, dim=-1).t() @ train_labels_one_hot) / (train_labels_one_hot.sum(0) + 1e-10)

    # Set the alpha constraint based on performance
    alpha_constraint = performance.to(device)
    print(f"Class-adaptive constraint weights (first 5 classes): {alpha_constraint[:5].detach().cpu().numpy()}")

    # Create optimizer for prototypes
    optimizer = torch.optim.SGD([prototypes], lr=args.clap_learning_rate if hasattr(args, 'clap_learning_rate') else 0.01, momentum=0.9)

    # Calculate repetition factor to match effective training steps with large datasets
    repeat_factor = 300  # Fixed to 60 to get 5 effective epochs over 300 total epochs
    print(f"Using repetition factor of {repeat_factor} for training")

    # Create a sampler that repeats the dataset
    class RepeatedSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, repeats=1):
            self.dataset = dataset
            self.repeats = repeats

        def __iter__(self):
            for _ in range(self.repeats):
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
                for idx in indices:
                    yield idx

        def __len__(self):
            return len(self.dataset) * self.repeats

    # Create data loader with repeated samples
    train_loader_repeated = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RepeatedSampler(train_dataset, repeats=repeat_factor),
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )
    print(f"Training with RepeatedSampler: {repeat_factor} repeats of the dataset")

    # Adjust number of epochs based on repetition
    effective_epochs = 1  # Fixed to 5 effective epochs
    print(f"Training for {effective_epochs} effective epochs")

    # Create scheduler
    total_steps = effective_epochs * len(train_loader_repeated)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
        )

    # Training loop
    print("Training CLAP teacher model...")
    val_freq = max(1, min(500, total_steps // 20))  # Validate every ~5% of training or every 500 steps, whichever is smaller
    val_acc = 0.0
    step_counter = 0

    # Main training loop
    for epoch in range(effective_epochs):
        # Set model to eval mode (only prototypes are trained)
        clip_model.eval()

        # Process batches
        pbar = tqdm(train_loader_repeated, desc=f"Epoch {epoch+1}/{effective_epochs}", mininterval=0.0, miniters=50)

        for images, targets in pbar:
            step_counter += 1
            images = images.to(device)
            targets = targets.to(device)

            # Apply teacher normalization
            images = teacher_normalize(images)

            # Extract features using CLIP model (with no grad)
            with torch.no_grad():
                features = clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)

            # Training step
            optimizer.zero_grad()

            # Normalize prototypes
            prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)

            # Calculate logits and loss
            logit_scale = clip_model.logit_scale.exp()
            logits = features @ prototypes_norm * logit_scale

            # Cross-entropy loss
            loss_ce = F.cross_entropy(logits, targets)

            # Constraint loss - L2 distance weighted by class-adaptive alphas
            diffs = (prototypes.t() - clip_weights_before_norm.t().clone()).pow(2).sum(-1)
            constraint_weight = args.clap_constraint_weight if hasattr(args, 'clap_constraint_weight') else 1.0
            loss_constraint = torch.mean(alpha_constraint * diffs)

            # Total loss
            loss = loss_ce + constraint_weight * loss_constraint

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update tqdm description
            lr = optimizer.param_groups[0]['lr']
            if step_counter % 50 == 0:
                pbar.set_description(f"Epoch {epoch+1}/{effective_epochs} | CE: {loss_ce.item():.3f} | Constr: {loss_constraint.item():.3f} | LR: {lr:.1e}")

            # Perform validation every val_freq steps for logging purposes only
            if step_counter % val_freq == 0 or (epoch == effective_epochs - 1 and step_counter == len(train_loader_repeated)):
                with torch.no_grad():
                    # Normalize prototypes
                    curr_prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)

                    # Compute logits
                    logit_scale = clip_model.logit_scale.exp()
                    val_logits = val_features @ curr_prototypes_norm * logit_scale

                    # Compute accuracy (only for logging)
                    val_acc = (val_logits.argmax(dim=1) == val_targets).float().mean().item() * 100

                    pbar.set_description(f"Epoch {epoch+1}/{effective_epochs} | CE: {loss_ce.item():.3f} | Constr: {loss_constraint.item():.3f} | Val: {val_acc:.2f}% | LR: {lr:.1e}")

                    # Save last checkpoint (always override)
                    torch.save({
                        'prototypes': prototypes.detach().clone(),
                        'epoch': epoch,
                        'step': step_counter,
                        'val_acc': val_acc,
                    }, last_ckpt_path)

    # Final evaluation on test set (for logging purposes only)
    prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)
    test_acc = evaluate_clap(clip_model, teacher_test_loader, teacher_normalize, prototypes_norm,
                             precomputed_features=test_features, precomputed_targets=test_targets)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Create model dict to return
    clap_dict = {
        'prototypes': prototypes.detach().clone(),
        'test_acc': test_acc,
        'val_acc': val_acc,
    }

    # Move CLIP model back to the original device for student training
    if args.local_rank != -1:
        clip_model = clip_model.to(f"cuda:{args.local_rank}")

    return clap_dict

def evaluate_clap(clip_model, data_loader, normalize, prototypes, precomputed_features=None, precomputed_targets=None):
    """
    Evaluate CLAP model on a dataset.

    Args:
        clip_model: CLIP model
        data_loader: DataLoader for test data
        normalize: Normalization transform
        prototypes: Prototype vectors
        precomputed_features: Optional precomputed image features
        precomputed_targets: Optional precomputed targets

    Returns:
        Accuracy as a percentage
    """
    correct = 0
    total = 0

    # Determine the device to use - should match the device of the provided prototypes
    device = prototypes.device

    with torch.no_grad():
        if precomputed_features is not None and precomputed_targets is not None:
            # Use precomputed features for faster evaluation
            features = precomputed_features
            targets = precomputed_targets

            # Normalize prototypes
            prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)

            # Compute logits
            logit_scale = clip_model.logit_scale.exp()
            logits = features @ prototypes_norm * logit_scale

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
        else:
            # Compute features on-the-fly
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images = images.to(device)
                targets = targets.to(device)

                # Extract and normalize features
                images = normalize(images)
                features = clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)

                # Normalize prototypes
                prototypes_norm = prototypes / prototypes.norm(dim=0, keepdim=True)

                # Compute logits
                logit_scale = clip_model.logit_scale.exp()
                logits = features.float() @ prototypes_norm.float() * logit_scale

                # Compute accuracy
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

    return 100 * correct / total
