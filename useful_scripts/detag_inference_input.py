import json

from load_helpers import remove_span_tags


# Function to load and process the infer.json file
def process_inference_data(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Apply remove_span_tags to all strings in the array
    processed_data = [remove_span_tags(text) for text in data]

    # Save the processed data back to the same file
    with open(file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)


# Call the function with the path to your infer.json file
process_inference_data('../infer/toy_train.json')
