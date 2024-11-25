from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import groq as Groq
model = SentenceTransformer('all-mpnet-base-v2')
abbreviations = {
    "Inc.": "Incorporated",
    "Ltd.": "Limited",
    "LLC": "Limited Liability Company",
    "PLC": "Public Limited Company",
    "Corp.": "Corporation",
    "Co.": "Company",
    "S.A.": "Société Anonyme (French for public company)",
    "Pvt. Ltd.": "Private Limited",
    "R&D": "Research and Development",
    "CEO": "Chief Executive Officer",
    "COO": "Chief Operating Officer",
    "CFO": "Chief Financial Officer",
    "CTO": "Chief Technology Officer",
    "VP": "Vice President",
    "SVP": "Senior Vice President",
    "EVP": "Executive Vice President",
    "HR": "Human Resources",
    "IT": "Information Technology",
    "PR": "Public Relations",
    "QA": "Quality Assurance",
    "RA": "Regulatory Affairs",
    "SaaS": "Software as a Service",
    "PaaS": "Platform as a Service"
}


def expand_company_name(company_name: str, abbreviation_dict: dict) -> str:
    normalized_dict = {key.lower().rstrip(string.punctuation): value for key, value in abbreviation_dict.items()}

    # Split company name into words
    words = company_name.split()

    # Expand abbreviations
    expanded_words = [
        normalized_dict.get(word.lower().rstrip(string.punctuation), word) for word in words
    ]

    return ' '.join(expanded_words)


def get_suggestions(name):
    client = Groq.Client(
        api_key="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",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for providing standerdized company names suggestions. Your tasks include: "
                "1. Make sure your provided sugestion does not exists or simmilar to the following list: "
                "'Bluewave Solutions Ltd', 'SwiftPeak Technologies Inc', 'Evergreen Ventures LLC', 'CrystalEdge Innovations', "
                "'Lunar Horizon Enterprises', 'Silverline Analytics', 'Brightpath Consulting Group', 'Vortex Dynamics Inc', "
                "'Aurora Nexus Holdings', 'Ironclad Security Systems', 'Crestline Developers Limited', 'Nexon BioTech LLC', "
                "'VelvetStream Apparel Co', 'UrbanOrbit Interiors', 'Pioneer Energy Solutions', 'StellarCore Technologies', "
                "'MapleCrest Foods LLC', 'Quantum Ridge Consulting', 'Oceanic Breeze Travel Co', 'Timberland Industrial Supplies', "
                "'Zenith Health Partners', 'AmberField Construction Group', 'NextEra EcoWorks', 'Fusion Point Media', "
                "'Skyline Horizons Inc', 'Four Seasons'. "
                "provide just 3 suggestion in bullet points like sutructure, dont need any explanation just need 3 suggestion "
            )

        },

        {
            "role": "user",
            "content": f"{name}"
        }
    ]

    try:
        # Call Groq's chat completion API
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",  # Replace with the desired model available in Groq
            max_tokens=50,
            temperature=1
        )
        standardized_name = response.choices[0].message.content.strip()
        return standardized_name
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return name  # Fall back to the original name if API call fails


def extract_suggestions(text: str) -> list:
    """
    Extracts three company name suggestions from the provided text.

    :param text: The input text containing suggestions.
    :return: A list of three suggestions.
    """
    # Split the text into lines and filter lines containing suggestions
    lines = text.split('\n')
    suggestions = [line.strip("• ").strip() for line in lines if "•" in line]
    return suggestions[:3]  # Return only the first three suggestions

prohibited_words = ["bank", "police", "government","unauthorized"]
def get_similarities(new_name, existing_names):
    if new_name == "":
        return ""

    if new_name.lower() in [name.lower() for name in existing_names]:
        return "Name exists in database"

    for word in prohibited_words:
        if word.lower() in new_name.lower():
            return "Name contains prohibited word."

    # Standardize new name and existing names using LLM
    standardized_new_name = expand_company_name(new_name,abbreviations)
    standardized_existing_names = [expand_company_name(name,abbreviations) for name in existing_names]
    print(standardized_new_name)
    print(standardized_existing_names)
    # Embed standardized names
    all_names = standardized_existing_names + [standardized_new_name]
    embeddings = model.encode(all_names)

    # Compute cosine similarity
    new_name_embedding = embeddings[-1].reshape(1, -1)
    existing_embeddings = embeddings[:-1]
    similarities = cosine_similarity(new_name_embedding, existing_embeddings)[0]

    # Return list of similarities
    result = [(name, round(similarity,2)) for name, similarity in zip(existing_names, similarities)]
    result =  sorted(result, key=lambda x: x[1], reverse=True)[:2]

    if float(result[0][1]) >= 0.7:
         return (f"Relevant name exits: {result[0][0]}\nPercentage Score: {round(float(result[0][1]),2)}\n"
                 f"Suggestions:\n"
                 f"{extract_suggestions(get_suggestions(result[0][0]))[0]}\n"
                 f"{extract_suggestions(get_suggestions(result[0][0]))[1]}\n"
                 f"{extract_suggestions(get_suggestions(result[0][0]))[2]}\n")
    else:
        return ("Name can be used")





