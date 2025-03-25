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