import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to read the dataset from a text file
def read_dataset_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    entries = content.split('\n\n')
    dataset = {}
    for entry in entries:
        if entry.strip():
            lines = entry.split('\n')
            title = lines[0].strip().replace('=', '').strip()  # Remove '=' symbols and trim whitespace
            context = ' '.join(lines[1:]).strip()
            dataset[title] = context
    return dataset

# Function to extract individual nouns and verbs
def extract_words(text):
    doc = nlp(text)
    words = [
        token.lemma_
        for token in doc
        if (token.pos_ in ["NOUN", "VERB"])
        and (token.lemma_.lower() not in nlp.Defaults.stop_words)
        and token.is_alpha
    ]
    return words

# Function to create a vector database for words
def create_vector_database(words):
    vector_database = {}
    for word in words:
        vector_database[word] = nlp(word).vector
    return vector_database

# Function to filter similar words
def filter_similar_words(vector_database, threshold=0.95):
    words = list(vector_database.keys())
    vectors = np.array(list(vector_database.values()))
    similarity_matrix = cosine_similarity(vectors)

    to_remove = set()
    removed_pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if similarity_matrix[i, j] > threshold:
                to_remove.add(words[j])
                removed_pairs.append((words[i], words[j]))

    filtered_database = {word: vector for word, vector in vector_database.items() if word not in to_remove}
    return filtered_database, removed_pairs

# Function to filter similar titles and contexts
def filter_titles_and_contexts(dataset, threshold=0.95):
    titles = list(dataset.keys())
    title_vectors = [nlp(title).vector for title in titles]

    similarity_matrix = cosine_similarity(title_vectors)
    to_remove = set()
    removed_pairs = []

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if similarity_matrix[i, j] > threshold:
                to_remove.add(titles[j])
                removed_pairs.append((titles[i], titles[j]))

    filtered_dataset = {title: context for title, context in dataset.items() if title not in to_remove}
    return filtered_dataset, removed_pairs

# Function to save filtered dataset to a CSV file
def save_to_csv(filtered_dataset, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for title, context in filtered_dataset.items():
            writer.writerow({'Title': title, 'Context': context})

# Path to the text file
file_path = '/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/trainDir/datasets/babylm_100M/simple_wiki.train'

# Read the dataset from the text file
simple_wiki_dataset = read_dataset_from_txt(file_path)

# Process the dataset until 10,000,000 tokens or all entries are processed
total_tokens = 0
filtered_dataset = {}
for title, context in simple_wiki_dataset.items():
    # Extract words from context
    words = extract_words(context)
    total_tokens += len(words)

    # Check if adding this entry exceeds token limit
    if total_tokens > 10000000:
        print(f"Token limit reached. Total tokens: {total_tokens}")
        break

    # Add title and context to filtered dataset
    filtered_dataset[title] = context

# Filter similar titles and contexts
filtered_dataset, removed_pairs = filter_titles_and_contexts(filtered_dataset)

# Print the filtered titles and contexts
#print("Filtered Titles and Contexts:")
#for title, context in filtered_dataset.items():
#    print(f"Title: {title}\nContext: {context}\n")

# Print the omitted titles
#print("\nOmitted Titles:")
#for title1, title2 in removed_pairs:
#    print(f"Removed '{title2}' because it's similar to '{title1}'")

# Save filtered dataset to CSV file
output_file = 'filtered_dataset.csv'
save_to_csv(filtered_dataset, output_file)
print(f"\nFiltered dataset saved to {output_file}")
