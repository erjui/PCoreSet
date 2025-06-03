import os

import torch
import torch.nn.functional as F
import transformers
from torchvision import transforms
from tqdm import tqdm

from datasets.utils import build_data_loader


def evaluate_clap(clip_model, data_loader, normalize, prototypes, precomputed_features=None, precomputed_targets=None):
    """
    Evaluate CLAP model on a dataset (test set only).
    Note: This function is used for reporting purposes only, not for model selection.

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
            # Compute features on-the-fly (slower)
            for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
                images = images.cuda()
                targets = targets.cuda()

                # Extract and normalize features
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

def train_clap_teacher(args, clip_model, dataset, clip_weights_before_norm, teacher_normalize, round=0):
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
        dataset: Dataset with train_x, val, and test splits
        clip_weights_before_norm: CLIP text weights before normalization
        teacher_normalize: Normalization transform for teacher (CLIP) model
        round: Current active learning round

    Returns:
        Dict with the trained CLAP model parameters
    """
    print("\n" + "=" * 60)
    print(f"🚀 TRAINING CLAP TEACHER (ROUND {round})")
    print("=" * 60)
    print(f"📚 Training with {len(dataset.train_x)} labeled samples")

    # Create directory for teacher models
    save_dir = os.path.join(args.log_dir, args.active_learning_strategy, str(round), "teacher")
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, f"clap_teacher_{args.shots}shots_round{round}.pt")
    # Add path for last checkpoint
    last_ckpt_path = os.path.join(save_dir, f"clap_teacher_last_{args.shots}shots_round{round}.pt")

    device = clip_model.visual.conv1.weight.device

    # Define custom augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Initialize model parameters with CLIP weights before normalization
    print("Initializing CLAP prototypes with CLIP weights before normalization")
    prototypes = clip_weights_before_norm.clone().detach()
    prototypes.requires_grad = True

    # Calculate normalized clip_weights for evaluation and constraints
    clip_weights = clip_weights_before_norm.clone()
    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)

    # Hyperparameters for CLAP
    alpha_constraint = torch.zeros(clip_weights.shape[1]).to(device)
    penalty_parameter = torch.zeros_like(alpha_constraint).to(device)  # For adaptive constraint

    # Create data loaders
    val_loader = build_data_loader(
        dataset.val, batch_size=args.batch_size,
        tfm=val_transform, is_train=False,
    )

    test_loader = build_data_loader(
        dataset.test, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )

    # Precompute validation features for efficient evaluation
    print("Precomputing validation features for efficient evaluation...")
    val_features = []
    val_targets = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Precomputing validation features", leave=False):
            images = images.cuda()
            targets = targets.cuda()
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            val_features.append(features)
            val_targets.append(targets)

    # Concatenate all batches
    val_features = torch.cat(val_features, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    print(f"Precomputed features for {val_features.size(0)} validation samples")

    # Precompute test features for final evaluation
    print("Precomputing test features for efficient evaluation...")
    test_features = []
    test_targets = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Precomputing test features", leave=False):
            images = images.cuda()
            targets = targets.cuda()
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            test_features.append(features)
            test_targets.append(targets)

    # Concatenate all batches
    test_features = torch.cat(test_features, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    print(f"Precomputed features for {test_features.size(0)} test samples")

    with torch.no_grad():
        # Evaluate zero-shot CLIP accuracy
        zeroshot_acc = evaluate_clap(clip_model, test_loader, teacher_normalize, clip_weights,
                                           precomputed_features=test_features, precomputed_targets=test_targets)
        print(f"Zero-shot Accuracy: {zeroshot_acc:.2f}%")

    # Calculate repetition factor
    repeat_factor = 300

    # Create a data loader that samples with replacement
    train_loader_labeled_repeated = build_data_loader(
        dataset.train_x, batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(
            dataset.train_x,
            replacement=True,
            num_samples=len(dataset.train_x) * repeat_factor
        ),
        tfm=train_transform, is_train=True, shuffle=False, drop_last=True, duplicate_if_needed=True
    )

    # Create a non-augmented dataloader for constraint calculation
    train_loader_no_aug = build_data_loader(
        dataset.train_x, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )

    # Adjust number of epochs
    effective_epochs = max(1, args.clap_epochs // repeat_factor)
    print(f"Original epochs: {args.clap_epochs}, Effective epochs: {effective_epochs}")

    # Extract features from non-augmented training set (used for init_lagrangian_multipliers)
    print("Extracting features from non-augmented training set for constraint initialization...")
    train_features_no_aug, train_labels_no_aug = [], []
    with torch.no_grad():
        for images, targets in tqdm(train_loader_no_aug, desc="Computing class-adaptive constraints"):
            images = images.cuda()
            targets = targets.cuda()

            # Extract CLIP features without augmentation
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

            train_features_no_aug.append(features)
            train_labels_no_aug.append(targets)

        train_features_no_aug = torch.cat(train_features_no_aug)
        train_labels_no_aug = torch.cat(train_labels_no_aug)

        # Get zero-shot logits for training set
        train_zs_logits = 100. * train_features_no_aug @ clip_weights
        train_zs_logits = train_zs_logits.float()

    # Initialize lagrangian multipliers (matching the exact method in adapters.py)
    print("Getting initial lagrangian multipliers for constraint formulation")
    with torch.no_grad():
        # Get one-hot encoding ground-truth
        train_labels_one_hot = F.one_hot(train_labels_no_aug, num_classes=clip_weights.shape[1]).float()

        # Get zero_shot performance
        performance = torch.diag(torch.softmax(train_zs_logits, dim=-1).t() @ train_labels_one_hot) / (train_labels_one_hot.sum(0) + 1e-10)

    # Set the alpha constraint based on performance
    alpha_constraint = performance.to(device)
    print(f"Class-adaptive constraint weights (first 5 classes): {alpha_constraint[:5].detach().cpu().numpy()}")

    # Create optimizer for prototypes (using the exact same optimizer as in adapters.py)
    optimizer = torch.optim.SGD([prototypes], lr=0.01, momentum=0.9)

    # Create scheduler
    total_steps = effective_epochs * len(train_loader_labeled_repeated)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    print(f"Using linear warmup scheduler with {0.1 * total_steps:.0f} warmup steps out of {total_steps} total steps")

    # Tracking metrics
    train_losses = []
    train_accs = []
    val_accs = []
    lr_history = []

    # Training loop
    print("Training CLAP teacher model...")
    val_freq = max(1, total_steps // 50) if total_steps > 0 else 1
    total_steps = 0

    # Main training loop - matching adapters.py implementation
    for epoch in range(effective_epochs):
        # Set model to eval mode (only prototypes are trained)
        clip_model.eval()

        # Initialize epoch metrics
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        epoch_corrects = 0
        epoch_total = 0

        # Process batches
        pbar = tqdm(train_loader_labeled_repeated, desc=f"Epoch {epoch+1}/{effective_epochs}")
        for step, (images, targets) in enumerate(pbar):
            total_steps += 1
            images = images.cuda()
            targets = targets.cuda()

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

            # Constraint loss - exactly matching zero_shot_constraint in adapters.py
            diffs = (prototypes.t() - clip_weights_before_norm.t().clone()).pow(2).sum(-1)
            loss_constraint = torch.mean(alpha_constraint * diffs)

            # Total loss
            loss = loss_ce + loss_constraint

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update metrics
            preds = logits.argmax(dim=1)
            batch_correct = (preds == targets).sum().item()
            batch_acc = 100 * batch_correct / targets.size(0)
            epoch_corrects += batch_correct
            epoch_total += targets.size(0)
            epoch_loss += loss.item() * targets.size(0)
            epoch_ce_loss += loss_ce.item() * targets.size(0)
            epoch_constraint_loss += loss_constraint.item() * targets.size(0)

            # Update tqdm description
            lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Epoch {epoch+1}/{effective_epochs} | CE: {loss_ce.item():.3f} | Constr: {loss_constraint.item():.3f} | Acc: {batch_acc:.2f}% | LR: {lr:.1e}")

            # Only perform validation every val_freq steps
            if total_steps % val_freq == 0:
                with torch.no_grad():
                    val_acc = evaluate_clap(clip_model, val_loader, teacher_normalize, prototypes,
                                            precomputed_features=val_features, precomputed_targets=val_targets)
                    val_accs.append(val_acc)

                    pbar.set_description(f"Epoch {epoch+1}/{effective_epochs} | CE: {loss_ce.item():.3f} | Constr: {loss_constraint.item():.3f} | Acc: {batch_acc:.2f}% | Val: {val_acc:.2f}% | LR: {lr:.1e}")

        # Calculate epoch metrics
        epoch_loss = epoch_loss / epoch_total
        epoch_ce_loss = epoch_ce_loss / epoch_total
        epoch_constraint_loss = epoch_constraint_loss / epoch_total
        epoch_acc = 100 * epoch_corrects / epoch_total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Save current learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Log training metrics
        print(f"Epoch {epoch+1}/{effective_epochs} - Loss: {epoch_loss:.4f}, CE Loss: {epoch_ce_loss:.4f}, Constraint Loss: {epoch_constraint_loss:.4f}, Acc: {epoch_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Evaluate final model on test set
    with torch.no_grad():
        # Evaluate CLAP model
        test_acc = evaluate_clap(clip_model, test_loader, teacher_normalize, prototypes,
                              precomputed_features=test_features, precomputed_targets=test_targets)
        print(f"CLAP Teacher Test Accuracy: {test_acc:.2f}%")

    # Save the last checkpoint (same as final for now)
    torch.save({
        'prototypes': prototypes,
        'clip_weights': clip_weights,
        'clip_weights_before_norm': clip_weights_before_norm,
        'init_mode': args.clap_init_mode,
        'test_acc': test_acc,
        'val_acc': val_accs[-1] if val_accs else None,
        'epoch': epoch
    }, last_ckpt_path)
    print(f"Saved last model checkpoint to {last_ckpt_path}")

    # Return trained model parameters
    result = {
        'prototypes': prototypes,
        'clip_weights': clip_weights,
        'clip_weights_before_norm': clip_weights_before_norm,
        'init_mode': args.clap_init_mode,
        'test_acc': test_acc,
        'val_acc': val_accs[-1] if val_accs else None,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    # Delete checkpoint file to save disk space after copying all necessary data
    if args.delete_checkpoint_after_use:
        if os.path.exists(last_ckpt_path):
            try:
                os.remove(last_ckpt_path)
                print(f"✅ Removed checkpoint file {last_ckpt_path} to save disk space")
            except Exception as e:
                print(f"❌ Failed to remove checkpoint file: {e}")

    return result
