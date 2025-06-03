import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Main function to select queries based on various strategies
def select_queries(args, query_pool, strategy='random', support_loader=None, student_normalize=None, student_model=None):
    """Selects samples to query based on the specified strategy"""
    all_indices = torch.cat([batch['indices'] for batch in query_pool])
    all_features = torch.cat([batch['features'] for batch in query_pool])
    all_ce_out = torch.cat([batch['ce_out'] for batch in query_pool])
    all_kd_out = torch.cat([batch['kd_out'] for batch in query_pool])
    all_teacher_logits = torch.cat([batch['teacher_logits'] for batch in query_pool])
    all_labels = torch.cat([batch['labels'] for batch in query_pool])

    #!. Use fixed alpha and beta (cannot access to validation set in Active Learning)
    best_alpha, best_beta = 0.5, 1.0

    # Log query selection start
    print(f"Starting query selection using {strategy} strategy")
    print(f"Query pool size: {len(all_indices)}")
    print(f"Queries to select: {args.active_learning_queries}")

    #################################
    # Baseline Selection Strategies
    #################################
    if strategy == 'random':
        # Random selection
        selected_indices = torch.randperm(len(all_indices))
        return all_indices[selected_indices[:args.active_learning_queries]]

    elif 'uncertainty' in strategy:
        ce_probs = F.softmax(all_ce_out, dim=1)
        kd_probs = F.softmax(all_kd_out / best_beta, dim=1)
        combined_probs = (1 - best_alpha) * ce_probs + best_alpha * kd_probs

        # Calculate uncertainty using both heads
        probs = combined_probs
        all_uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

        _, selected_idx = torch.sort(all_uncertainties, descending=True)

        # Get statistics for selected samples
        selected_uncertainties = all_uncertainties[selected_idx[:args.active_learning_queries]]
        print("\n" + "=" * 60)
        print(f"📊 UNCERTAINTY SCORE STATISTICS ({strategy})")
        print("=" * 60)
        print(f"Uncertainty scores: mean={selected_uncertainties.mean():.4f}, std={selected_uncertainties.std():.4f}")
        print("=" * 60 + "\n")

        return all_indices[selected_idx[:args.active_learning_queries]]

    elif strategy == 'coreset':
        # Use features for coreset selection using k-center-greedy algorithm
        labeled_features = []
        with torch.no_grad():
            for images, _ in support_loader:
                images = images.cuda()

                # Get features from both heads
                _, _, features = student_model(student_normalize(images))

                labeled_features.append(features.cpu())
        labeled_features = torch.cat(labeled_features, dim=0)

        # Normalize features
        all_features = F.normalize(all_features, dim=1)
        labeled_features = F.normalize(labeled_features, dim=1)

        # Initialize arrays for k-center-greedy selection
        selected_indices = []
        remaining_indices = list(range(len(all_features)))

        # Track diversity scores
        diversity_scores = []

        # Calculate initial distances to labeled set
        # Shape: [n_unlabeled, n_labeled]
        # Move features to GPU for faster computation
        all_features_gpu = all_features.cuda()
        labeled_features_gpu = labeled_features.cuda()

        distances_to_labeled = 1 - torch.mm(all_features_gpu, labeled_features_gpu.t())
        min_distances, _ = torch.min(distances_to_labeled, dim=1)

        # Select the first point (furthest from labeled set)
        first_idx = torch.argmax(min_distances).item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        diversity_scores.append(min_distances[first_idx].item())

        # Create a GPU tensor for selected features
        selected_features_gpu = all_features_gpu[first_idx].unsqueeze(0)

        # Iteratively select points
        for _ in tqdm(range(args.active_learning_queries - 1), desc='K-center-greedy selection'):
            if len(remaining_indices) == 0:
                break

            # Process in batches to avoid memory issues
            batch_size = min(1000, len(remaining_indices))
            current_min_distances = torch.zeros(len(remaining_indices), device='cuda')

            for i in range(0, len(remaining_indices), batch_size):
                batch_indices = remaining_indices[i:i+batch_size]
                batch_features = all_features_gpu[batch_indices]

                # Calculate distances to selected centers
                # Shape: [batch_size, n_selected]
                distances_to_selected = 1 - torch.mm(batch_features, selected_features_gpu.t())
                min_distances_to_selected, _ = torch.min(distances_to_selected, dim=1)

                # Get distances to labeled set for batch indices
                min_distances_to_labeled, _ = torch.min(distances_to_labeled[batch_indices], dim=1)

                # Take minimum distance to either labeled or selected points
                current_min_distances[i:i+len(batch_indices)] = torch.min(
                    min_distances_to_labeled,
                    min_distances_to_selected
                )

            # Select the point with maximum minimum distance
            max_min_idx = torch.argmax(current_min_distances).item()
            selected_idx = remaining_indices[max_min_idx]

            # Store diversity score
            diversity_scores.append(current_min_distances[max_min_idx].item())

            # Update selected indices and features
            selected_indices.append(selected_idx)
            selected_features_gpu = torch.cat([
                selected_features_gpu,
                all_features_gpu[selected_idx].unsqueeze(0)
            ], dim=0)

            remaining_indices.remove(selected_idx)

        # Move back to CPU after selection is complete
        all_features_gpu = all_features_gpu.cpu()
        labeled_features_gpu = labeled_features_gpu.cpu()
        selected_features_gpu = selected_features_gpu.cpu()

        # Print statistics for diversity scores
        diversity_scores = torch.tensor(diversity_scores)
        print("\n" + "=" * 60)
        print("📊 CORESET DIVERSITY SCORE STATISTICS")
        print("=" * 60)
        print(f"Diversity scores: mean={diversity_scores.mean():.4f}, std={diversity_scores.std():.4f}")
        print("=" * 60 + "\n")

        return all_indices[selected_indices[:args.active_learning_queries]]

    elif 'badge' in strategy:
        # BADGE: Batch Active learning by Diverse Gradient Embeddings
        print(f"Computing BADGE gradient embeddings with {strategy} strategy...")

        # In memory-saving mode, we don't have images, but we can use the pre-computed features and logits
        all_features = torch.cat([batch['features'] for batch in query_pool])
        num_samples = len(all_features)
        num_classes = all_ce_out.shape[1]

        # Use cached logits and features instead of recomputing
        gradient_embeddings = []
        with torch.no_grad():
            if 'cehead' in strategy:
                # Use only CE head probabilities
                probs = F.softmax(all_ce_out, dim=1)
                pred_classes = probs.argmax(dim=1)

                # Create one-hot encoding for predicted classes
                y_one_hot = torch.zeros_like(probs)
                y_one_hot.scatter_(1, pred_classes.unsqueeze(1), 1)

                # Compute (y - p) for all classes and samples at once
                diff = y_one_hot - probs  # Shape: [num_samples, num_classes]

                # Compute gradient embeddings using pre-computed features
                # Reshape features to [num_samples, 1, feature_dim]
                features_expanded = all_features.unsqueeze(1)
                # Reshape diff to [num_samples, num_classes, 1]
                diff_expanded = diff.unsqueeze(2)
                # Compute outer product: [num_samples, num_classes, feature_dim]
                embeddings = diff_expanded * features_expanded

                # Flatten
                gradient_embeddings = embeddings.view(embeddings.size(0), -1)

            elif 'kdhead' in strategy:
                # Use only KD head probabilities
                probs = F.softmax(all_kd_out, dim=1)
                pred_classes = probs.argmax(dim=1)

                # Create one-hot encoding for predicted classes
                y_one_hot = torch.zeros_like(probs)
                y_one_hot.scatter_(1, pred_classes.unsqueeze(1), 1)

                # Compute (y - p) for all classes and samples at once
                diff = y_one_hot - probs  # Shape: [num_samples, num_classes]

                # Compute gradient embeddings using pre-computed features
                # Reshape features to [num_samples, 1, feature_dim]
                features_expanded = all_features.unsqueeze(1)
                # Reshape diff to [num_samples, num_classes, 1]
                diff_expanded = diff.unsqueeze(2)
                # Compute outer product: [num_samples, num_classes, feature_dim]
                embeddings = diff_expanded * features_expanded

                # Flatten
                gradient_embeddings = embeddings.view(embeddings.size(0), -1)

            elif 'combined' in strategy or strategy == 'badge':
                # Apply the same interpolation as used for evaluation
                probs_ce = F.softmax(all_ce_out, dim=1)
                probs_kd = F.softmax(all_kd_out / best_beta, dim=1)
                combined_probs = (1 - best_alpha) * probs_ce + best_alpha * probs_kd

                # Get predicted classes for the batch using combined probabilities
                pred_classes = combined_probs.argmax(dim=1)

                # Create one-hot encoding for predicted classes
                y_one_hot = torch.zeros_like(combined_probs)
                y_one_hot.scatter_(1, pred_classes.unsqueeze(1), 1)

                # Compute (y - p) for all classes and samples at once
                diff = y_one_hot - combined_probs  # Shape: [num_samples, num_classes]

                # Compute gradient embeddings using pre-computed features
                # Reshape features to [num_samples, 1, feature_dim]
                features_expanded = all_features.unsqueeze(1)
                # Reshape diff to [num_samples, num_classes, 1]
                diff_expanded = diff.unsqueeze(2)
                # Compute outer product: [num_samples, num_classes, feature_dim]
                embeddings = diff_expanded * features_expanded

                # Flatten
                gradient_embeddings = embeddings.view(embeddings.size(0), -1)

        # Normalize embeddings
        gradient_embeddings = F.normalize(gradient_embeddings, dim=1)

        # Use k-means++ initialization to select diverse samples
        selected_indices = []

        # Track diversity scores
        diversity_scores = []

        # For BADGE, we should select the first point based on gradient embedding magnitudes
        gradient_norms = torch.norm(gradient_embeddings, dim=1)

        # Select first point with largest gradient norm
        first_idx = torch.argmax(gradient_norms).item()
        selected_indices.append(first_idx)
        diversity_scores.append(gradient_norms[first_idx].item())

        # Initialize mask for unselected samples
        mask = torch.ones(num_samples, dtype=torch.bool)
        mask[first_idx] = False

        # Select remaining points using k-means++ style selection with vectorized operations
        for _ in tqdm(range(args.active_learning_queries - 1), desc=f'{strategy} selection'):
            if not torch.any(mask):  # Break if all points are selected
                break

            # Compute distances to all selected points at once
            selected_embeddings = gradient_embeddings[selected_indices]
            distances = 1 - torch.mm(gradient_embeddings, selected_embeddings.t())

            # Get minimum distance to any selected point
            min_distances = torch.min(distances, dim=1)[0]

            # Set distances of already selected points to -inf
            min_distances[~mask] = -float('inf')

            # Select the point with maximum distance
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)
            diversity_scores.append(min_distances[next_idx].item())
            mask[next_idx] = False

        # Print statistics for diversity scores
        diversity_scores = torch.tensor(diversity_scores)
        print("\n" + "=" * 60)
        print(f"📊 {strategy.upper()} DIVERSITY SCORE STATISTICS")
        print("=" * 60)
        print(f"Diversity scores: mean={diversity_scores.mean():.4f}, std={diversity_scores.std():.4f}")
        print("=" * 60 + "\n")

        return all_indices[selected_indices]

    elif strategy == 'classbalanced':
        # Class Balanced Selection
        print("Using combined predictions for class balancing")

        # Get predicted labels based on the specified head
        ce_probs = F.softmax(all_ce_out, dim=1)
        kd_probs = F.softmax(all_kd_out / best_beta, dim=1)
        predicted_probs = (1 - best_alpha) * ce_probs + best_alpha * kd_probs

        # Get predicted labels from probabilities
        predicted_labels = torch.argmax(predicted_probs, dim=1)

        # Get class distribution from existing labeled set
        labeled_class_counts = {}
        if support_loader is not None:
            # Extract labels from support set
            support_labels = []
            with torch.no_grad():
                for _, labels in support_loader:
                    support_labels.append(labels)

            if support_labels:
                support_labels = torch.cat(support_labels)
                # Count occurrences of each class
                for label in support_labels:
                    label_idx = label.item()
                    labeled_class_counts[label_idx] = labeled_class_counts.get(label_idx, 0) + 1

        # Organize samples by predicted class
        per_class_indices = {}
        for class_idx in torch.unique(predicted_labels):
            # Get indices where prediction matches this class
            class_mask = (predicted_labels == class_idx)
            if not torch.any(class_mask):
                continue

            # Store indices for this class
            per_class_indices[class_idx.item()] = {
                'indices': torch.where(class_mask)[0],
                'count': labeled_class_counts.get(class_idx.item(), 0)  # Current count in labeled set
            }

        # Calculate inverse frequency weights for each class
        total_labeled = sum(labeled_class_counts.values()) if labeled_class_counts else len(per_class_indices)
        class_weights = {}

        # If we have labeled data, use inverse frequency weighting
        if labeled_class_counts:
            # Add small constant to avoid division by zero
            epsilon = 1.0
            for class_idx in per_class_indices:
                count = labeled_class_counts.get(class_idx, epsilon)
                # Inverse frequency weighting
                class_weights[class_idx] = total_labeled / (count + epsilon)
        else:
            # If no labeled data yet, use uniform weights
            for class_idx in per_class_indices:
                class_weights[class_idx] = 1.0

        # Normalize weights to sum to 1
        weight_sum = sum(class_weights.values())
        for class_idx in class_weights:
            class_weights[class_idx] /= weight_sum

        # Allocate queries per class based on weights
        queries_per_class = {}
        remaining_queries = args.active_learning_queries

        # First pass: allocate based on weights
        for class_idx, weight in class_weights.items():
            # Allocate queries proportional to weight
            queries_per_class[class_idx] = int(args.active_learning_queries * weight)
            remaining_queries -= queries_per_class[class_idx]

        # Second pass: distribute remaining queries to classes with highest weights
        if remaining_queries > 0:
            sorted_classes = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            for class_idx, _ in sorted_classes:
                if remaining_queries <= 0:
                    break
                queries_per_class[class_idx] += 1
                remaining_queries -= 1

        # Select samples randomly per class
        selected_idx = []
        remaining_queries = args.active_learning_queries

        print("\n" + "=" * 60)
        print("📊 CLASS BALANCING ALLOCATION (RANDOM SELECTION)")
        print("=" * 60)
        print(f"Class ID: Current Count -> Allocated Queries (Weight)")

        for class_idx, class_data in per_class_indices.items():
            class_indices = class_data['indices']
            current_count = class_data['count']
            allocated = queries_per_class.get(class_idx, 0)

            print(f"Class {class_idx}: {current_count} -> +{allocated} ({class_weights.get(class_idx, 0):.3f})")

            # Randomly shuffle indices for this class
            perm = torch.randperm(len(class_indices))
            shuffled_indices = class_indices[perm]

            # Select random queries for this class
            num_to_select = min(allocated, len(shuffled_indices), remaining_queries)
            selected_idx.extend(shuffled_indices[:num_to_select].tolist())
            remaining_queries -= num_to_select

        # If we still have queries to fill, take randomly from remaining samples
        if remaining_queries > 0:
            print(f"\nAllocating {remaining_queries} remaining queries randomly")
            all_remaining = []
            for class_idx, class_data in per_class_indices.items():
                class_indices = class_data['indices']

                # Skip indices we've already selected
                for i in class_indices:
                    if i.item() not in selected_idx:
                        all_remaining.append(i.item())

            # Randomly shuffle remaining indices
            import random
            random.shuffle(all_remaining)
            selected_idx.extend(all_remaining[:remaining_queries])

        selected_idx = torch.tensor(selected_idx)

        print("\n" + "=" * 60)
        print("📊 RANDOM SELECTION STATISTICS (PER CLASS)")
        print("=" * 60)
        print(f"Number of classes represented: {len(per_class_indices)}")
        for class_idx, class_data in per_class_indices.items():
            print(f"Class {class_idx}: samples={len(class_data['indices'])}")

        return all_indices[selected_idx[:args.active_learning_queries]]

    #################################
    # PCoreSet Selection Strategies
    #################################
    if 'pcoreset' in strategy:
        # P-Coreset selection
        print(f"Running P-Coreset selection for {args.active_learning_queries} queries")

        ce_probs = F.softmax(all_ce_out, dim=1)
        kd_probs = F.softmax(all_kd_out / best_beta, dim=1)
        selection_probs = (1 - best_alpha) * ce_probs + best_alpha * kd_probs
        print("Using combined probabilities for P-Coreset selection")

        # Get model predictions from labeled set instead of one-hot encoded labels
        labeled_probs = []

        with torch.no_grad():
            student_model.eval()
            for images, _ in support_loader:
                images = images.cuda()
                images = student_normalize(images)
                outputs_ce, outputs_kd, _ = student_model(images)

                # Apply the same probability calculation based on strategy
                probs_ce = F.softmax(outputs_ce, dim=1)
                probs_kd = F.softmax(outputs_kd / best_beta, dim=1)
                probs = (1 - best_alpha) * probs_ce + best_alpha * probs_kd
                labeled_probs.append(probs.cpu())


        # Use existing labeled data as starting points or pick a random starting point
        selected_indices = []
        labeled_probs = torch.cat(labeled_probs, dim=0)
        selected_probs = labeled_probs.clone()

        # Compute pairwise distances for remaining points
        remaining_indices = list(set(range(len(all_indices))) - set(selected_indices))

        # Move relevant tensors to GPU for faster computation
        selection_probs = selection_probs.cuda()
        selected_probs = selected_probs.cuda()

        # Select the rest of the points
        with torch.no_grad():
            for _ in tqdm(range(args.active_learning_queries), desc="Selecting diverse samples"):
                if not remaining_indices:
                    break

                # Calculate distances from each remaining point to all selected points
                min_distances = []

                # Process in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(remaining_indices), batch_size):
                    batch_indices = remaining_indices[i:i+batch_size]
                    batch_probs = selection_probs[batch_indices].cuda()
                    # Calculate pairwise distances between batch and selected points
                    distances = torch.cdist(batch_probs, selected_probs, p=2)

                    # Get minimum distance to any selected point
                    min_dist, _ = torch.min(distances, dim=1)
                    min_distances.append(min_dist.cpu())  # Move back to CPU for concatenation

                # Combine all batches
                min_distances = torch.cat(min_distances)

                # Select the point with maximum minimum distance (classic coreset selection)
                max_idx = torch.argmax(min_distances).item()
                next_idx = remaining_indices[max_idx]

                # Update selected and remaining indices
                selected_indices.append(next_idx)
                selected_probs = torch.cat([selected_probs, selection_probs[next_idx].unsqueeze(0).cuda()])
                remaining_indices.remove(next_idx)

        # Move back to CPU before returning
        selection_probs = selection_probs.cpu()
        selected_probs = selected_probs.cpu()

        print(f"P-Coreset selection complete. Selected {len(selected_indices)} samples.")
        return all_indices[selected_indices]

    # Default to random selection if no strategy matches
    print(f"Using default random selection for {args.active_learning_queries} queries")
    selected_indices = torch.randperm(len(all_indices))
    return all_indices[selected_indices[:args.active_learning_queries]]
