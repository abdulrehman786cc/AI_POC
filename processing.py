from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import openai

model = SentenceTransformer('all-mpnet-base-v2')
openai.api_key = "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"

def extract_company_name(response_text):
    match = re.search(r'"(.*?)"', response_text)
    return match.group(1) if match else response_text

def standardize_name_with_llm(name):
    messages = [
        {"role": "system",
         "content": "ou are a helpful assistant for standardizing company names. Your tasks include:1. Resolving abbreviations in company names (e.g., 'Pvt' becomes 'Private', 'Ltd' becomes 'Limited').2. Converting digits in company names into their word equivalents (e.g., '4 Tech' becomes 'Four Tech')."},
        {"role": "user", "content": f"{name}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use a suitable model, like gpt-4 or gpt-3.5-turbo
            messages=messages,
            max_tokens=50,
            temperature=0.2
        )
        standardized_name = response['choices'][0]['message']['content'].strip()
        return standardized_name
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return name  # Fall back to the original name if API call fails


prohibited_words = ["army", "bank", "police", "government","unauthorized"]
def get_similarities_with_llm(new_name, existing_names):
    if new_name=="":
        return ""
    if new_name.lower() in [name.lower() for name in existing_names]:
        return "Name exists in database"

    for word in prohibited_words:
        if word.lower() in new_name.lower():
            return "Name contains prohibited word."

    # Standardize new name and existing names using LLM
    standardized_new_name = extract_company_name(standardize_name_with_llm(new_name))
    standardized_existing_names = [extract_company_name(standardize_name_with_llm(name)) for name in existing_names]
    # Embed standardized names
    all_names = standardized_existing_names + [standardized_new_name]
    embeddings = model.encode(all_names)

    # Compute cosine similarity
    new_name_embedding = embeddings[-1].reshape(1, -1)
    existing_embeddings = embeddings[:-1]
    similarities = cosine_similarity(new_name_embedding, existing_embeddings)[0]

    # Return list of similarities
    result = [(name, round(similarity, 2)) for name, similarity in zip(existing_names, similarities)]
    result =  sorted(result, key=lambda x: x[1], reverse=True)[:2]
    if float(result[0][1]) >= 0.5:
         return (f"Relevant name exits: {result[0][0]} , Percentage Score: {round(float(result[0][1]),2)}")
    else:
        return ("Name can be used")





