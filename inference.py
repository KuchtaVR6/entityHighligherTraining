# def inference(model, val_dataset, label_map):
#     """Makes predictions on the validation dataset and prints the labels."""
#     model.eval()  # Set the model in evaluation mode
#     all_predictions = []
#
#     # Loop through the validation dataset
#     for batch in val_dataset:
#         # Get the input IDs and attention masks from the batch
#         input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)  # Add batch dimension
#         attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)
#
#         with torch.no_grad():
#             # Run the model to get logits (do not pass labels during inference)
#             outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits  # Logits are the second output
#
#             # Get the predicted labels by taking the argmax of the logits
#             predictions = torch.argmax(logits, dim=-1).squeeze().tolist()  # Remove batch dimension
#
#             # Map predictions back to label names
#             predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]
#             all_predictions.append(predicted_labels)
#
#         break
#
#     # Print the predicted labels
#     for predicted_labels in all_predictions:
#         print(predicted_labels)
#
#
# def compute_accuracy(model, val_dataset, label_map):
#     model.eval()  # Set the model in evaluation mode
#     all_predictions = []
#     correct_counts = defaultdict(int)
#     total_counts = defaultdict(int)
#     id_to_label = {v: k for k, v in label_map.items()}
#
#     for batch in tqdm(val_dataset):
#         # Get the input IDs and attention masks from the batch
#         input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)  # Add batch dimension
#         attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)
#
#         # Get ground truth labels
#         labels = batch["labels"]
#
#         with torch.no_grad():
#             # Run the model to get logits (do not pass labels during inference)
#             outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits  # Logits are the second output
#
#             # Get the predicted labels by taking the argmax of the logits
#             predictions = torch.argmax(logits, dim=-1).squeeze().tolist()  # Remove batch dimension
#
#             # Map predictions back to label names
#             predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]
#             all_predictions.append(predicted_labels)
#
#             # Calculate accuracy for each label
#             for i, label_id in enumerate(labels):
#                 if i >= len(predictions):  # Handle padding or truncated sequences
#                     break
#                 pred_label_id = predictions[i]
#
#                 # Count correct predictions
#                 if label_id == pred_label_id:
#                     correct_counts[label_id] += 1
#                 total_counts[label_id] += 1
#
#         break
#
#     # Calculate per-class accuracy
#     per_class_accuracy = {
#         id_to_label[class_id]: correct_counts[class_id] / total_counts[class_id]
#         for class_id in total_counts if total_counts[class_id] > 0
#     }
#
#     # Calculate overall accuracy
#     total_correct = sum(correct_counts.values())
#     total_samples = sum(total_counts.values())
#     overall_accuracy = total_correct / total_samples
#
#     per_class_accuracy["overall_accuracy"] = overall_accuracy
#
#     return per_class_accuracy