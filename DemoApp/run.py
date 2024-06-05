import os
import json
import openai
import spacy
import requests

from chromadb import Client
from typing import Optional
from langchain.llms import OpenAI
from langchain.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.agents import initialize_agent, AgentType, Tool

#SELECT clause prompt
select_template = """You are an expert in constructing SPARQL queries that extract information from Wikidata for the natural language query provided using the items and properties given to you.
Construct a SPARQL query (with SELECT clause) using the given Wikidata item and property IDs for the natural language query given referring to the examples given below. The final answer should be the SPARQL query
which would extract the answer to the original question from Wikidata when run by the user.

\n
Example 1:
Query: "Where was Joe Biden born?"
SPARQL query:
SELECT ?placeOfBirth WHERE {{
  wd:Q6279 wdt:P19 ?placeOfBirth .
}}
\n
Example 2:
Query: "List all people born in Chile."
SPARQL query:
SELECT?item?itemLabel
WHERE {{
?item wdt:P31 wd:Q5. # instance of human
?item wdt:P19 wd:Q16. # born in Chile
 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
\n
Example 3:
Query: "Who is the child of Ranavalona I's husband?"
SPARQL query:
SELECT ?itemName WHERE {{ wd:Q169794 wdt:P26 ?X .
                      ?X wdt:P22 ?item .
                      ?item rdfs:label ?itemName .
                      FILTER (lang(?itemName) = "en")
}}
\n
Example 4: Filter type queries
Query: "What award did Olga Tokarczuk receive in 2015?"
SPARQL query:
SELECT ?obj WHERE {{ wd:Q254032 p:P166 ?s . ?s ps:P166 ?obj . ?s pq:P585 ?x filter(contains(YEAR(?x),'2015')) }}
\n\n
You may assume the following prefixes:
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>

\n\n
The final answer should only be a SPARQL query with the ASK WHERE clause. Do not use prefixes other than the ones mentioned above. No need to provide an explanation regarding the query generated.

Translate the natural language query to SPARQL given the following information.
Natural language query: {query}
Items extracted from the query with their ID: {item_name} and {item_id}
Properties identified: {prop_dict}
"""

#ask_where prompt
ask_template = """You are an expert in constructing SPARQL queries that extract information from Wikidata for the natural language query provided using the items and properties given to you.
Construct a SPARQL query referring to the examples given below where True or False type queries are in natural language and you need
to generate the corresponding SPARQL queries (ASK WHERE type) using the given Wikidata item and property IDs given to you that would answer the the original query when run by the user.

\n
Example 1:
Query: "Is the life expectancy of Indonesia 55.3528?"
SPARQL:
ASK WHERE {{ wd:Q252 wdt:P2250 ?obj filter(?obj = 55.3528) }}
\n
Example 2:
Query: "Is the total fertility rate of Algeria greater than 3.4284?"
SPARQL query:
ASK WHERE {{ wd:Q262 wdt:P4841 ?obj filter(?obj > 3.4284) }}
\n
Example 3:
Query: "Is it true that the maximum wavelength of sensitivity of the human eye equal to 700?"
SPARQL query:
ASK WHERE {{ wd:Q430024 wdt:P3737 ?obj filter(?obj = 700) }}
\n
Example 4:
Query: "Is it true that Kevin Costner owner of Fielders Stadium?"
SPARQL query:
ASK WHERE {{ wd:Q11930 wdt:P1830 wd:Q5447154 }}
\n\n
You may assume the following prefixes:
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
\n\n
The final answer should only be a SPARQL query with the ASK WHERE clause. Do not use prefixes other than the ones mentioned above. No need to provide an explanation regarding the query generated.

Translate the natural language query to SPARQL given the following information.
Natural language query: {query}
Items extracted from the query with their ID: {item_name} and {item_id}
Properties identified: {prop_dict}
"""

# Initialize the OpenAI API client

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a class to handle SPARQL query classification
class SPARQLQueryClassifier:
    def __init__(self, api_key):
        self.api_key = api_key

    def classify_query(self, query):
        # Use GPT-4o to classify the query
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing natural language queries and determining if they require an ASK WHERE or a SELECT clause in SPARQL."},
                {"role": "user", "content": f"Classify the following query as True/False (ASK WHERE) or information retrieval (SELECT clause): '{query}'. Give the final answer only as either 'SELECT clause' or 'ASK WHERE clause'. No need to provide an explanation."}
            ]
        )
        return response

# Custom SPARQL query construction tool
class SPARQLQueryConstructor(BaseTool):
    name = "sparql_query_constructor"
    description = "Construct a SPARQL query based on the given prompt and item/property information."

    def _run(self, query):
        prompt = query.strip()

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in constructing SPARQL queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0
        )
        return response

    def _acrunch_shortcut(self):
        return "Construct a SPARQL query"

# Initialize the LLM agent
def initialize_sparql_query_agent():
    llm = OpenAI(temperature=0)
    tool = SPARQLQueryConstructor()
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_tokens = 100
    )
    return agent

# Function to lookup entity or property IDs in Wikidata
def vocab_lookup(search: str, entity_type: str = "item") -> Optional[str]:
    # Define the Wikidata API endpoint
    url = "https://www.wikidata.org/w/api.php"

    # Set the search namespace based on the entity type
    if entity_type == "item":
        srnamespace = 0
    elif entity_type == "property":
        srnamespace = 120
    else:
        raise ValueError("entity_type must be either 'property' or 'item'")

    # Set up the request parameters
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "type": entity_type,
        "search": search,
        "uselang": "en",
    }

    # Send the request to Wikidata API
    response = requests.get(url, params=params)

    # Parse the response and extract entity or property ID
    if response.status_code == 200:
        data = response.json()
        if "search" in data and data["search"]:
            for entry in data["search"]:
                if entry["display"]["label"]["value"].lower() == search.lower():
                  return entry["id"]
            return data["search"][0]["id"]
        else:
            return f"I couldn't find any {entity_type} for '{search}'. Please rephrase your request and try again."
    else:
        return "Sorry, I got an error. Please try again."

# Load the English language model with dependencies and NER
nlp = spacy.load("en_core_web_sm")

# Function to identify entities from a prompt
def identify_entities(prompt):
    doc = nlp(prompt)

    # Extract entities and their labels
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    # Extract nouns and their dependencies
    nouns = []
    for token in doc:
        if token.pos_ == "NOUN":
            noun_deps = [child.text for child in token.children if child.dep_ in ["nmod", "compound"]]
            noun_text = token.text + " " + " ".join(noun_deps)
            noun_dep = token.dep_
            nouns.append((noun_text, token.pos_, noun_dep))

    # Combine entities, nouns with dependencies, and identified components
    attributes = entities + nouns

    return attributes

# Simple echo tool to satisfy the ZeroShotAgent requirement
@tool
def echo_tool(input_text: str) -> str:
    """Echoes the input text."""
    return input_text

def identify_items_props(query, entities):

    # Initialize the LLM agent
    llm = OpenAI(temperature=0, api_key=openai.api_key)  # Make sure you have set up OpenAI API key as an environment variable
    tools = [Tool(name="EchoTool", func=echo_tool, description="Echoes the input text")]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=6,
        early_stopping_method="force"
    )

    # Construct the prompt and query the agent
    prompt = f"""
    You are building a pipeline to generate SPARQL queries based on natural language query. Wikidata is used as the data source, in which data is stored as items and properties. Each item could have some properties linked to it and their associated values.
    Now, given a natural language query and some identified entities from the query, I need you to identify what could be the main item from the query, and what could be associated property. Based on your decision, I will fetch all properties of the associated item and then try to find the closest matching property.

    Now I will share an example.

    Natural Language Query : 'Who was the last king of France?'
    Identified Entities :
    France (GPE)
    king  (NOUN, nsubj)

    Solution : {{"Item" : "France" , "Properties" : ["rulers", "kings"]}}

    Now I want you to provide the solution for another query.

    Natural Language Query : {query}
    Identified Entities:
    {entities}

    I want only the solution as a response.
    """

    result = agent.run(prompt)

    return result

# Wrapper
def identify_item(prompt):

  props = identify_entities(prompt)

  # Convert properties to a formatted string
  prop_str = "\n".join(" ".join(prop) for prop in props)

  # obtained props
  print("Identified Entities: \n", prop_str)

  return identify_items_props(prompt, prop_str)

# Function to get property labels from Wikidata
def get_property_labels(item_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={item_id}&props=claims&format=json&languages=en"
    response = requests.get(url).json()

    properties = []
    for prop_id, claims in response["entities"][item_id]["claims"].items():
        prop_url = f"https://www.wikidata.org/wiki/Special:EntityData/{prop_id}.json"
        prop_response = requests.get(prop_url).json()
        try:
            prop_label = prop_response["entities"][prop_id]["labels"]["en"]["value"]
        except KeyError:
            prop_label = None
        properties.append({"id": prop_id, "label": prop_label})

    return properties

# Function to create the Chroma database and store property data
def create_property_db(item_id):
    property_data = get_property_labels(item_id)

    chroma_client = Client()
    try:
        collection = chroma_client.create_collection(f"{item_id}_properties")
    except:
        chroma_client.delete_collection(f"{item_id}_properties")
        collection = chroma_client.create_collection(f"{item_id}_properties")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    for prop in property_data:
        prop_id = prop["id"]
        prop_label = prop["label"]
        if prop_label:
            texts = [prop_label]
            embeddings = embedding_model.encode(texts)
            metadatas = [{"property_id": prop_id}]
            ids = [f"{prop_id}_{prop_label.replace(' ', '_')}"]
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings  # Pass the precomputed embeddings here
            )

    return collection

# Function to initialize the LLM agent and query for the right property
def query_property(collection, query, item_name):
    # Retrieve property IDs and labels from the Chroma collection
    results = collection.get()
    property_data = [(metadata["property_id"], doc) for doc, metadata in zip(results["documents"], results["metadatas"])]

    # Construct the property list for the prompt
    property_list = "\n".join([f"{prop_id}: {prop_label}" for prop_id, prop_label in property_data])

    # Construct the prompt and query the agent
    prompt = f"""Given the following list of properties for {item_name}:\n\n{property_list}\n\nIdentify the right property/properties and 
    their corresponding property ID to answer the query: '{query}'\n\nThe final answer should only be in the following format:
    {{'property1': PID1, 'property2': PID2}}. A dictionary of properties as keys and its ID as value. The length will depend on the number of properties identified by you."""

    response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in identifying the right Wikidata properties to answer a question related to a Wikidata item."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0
    )
    return response

def query_generator(query):

    os.environ["OPENAI_API_KEY"] = openai.api_key
    result = json.loads(identify_item(query))
    item_name = result["Item"]
    print("Item Name: ", item_name)

    item_id = vocab_lookup(item_name)
    print("Item ID: ", item_id)

    collection = create_property_db(item_id)

    result = query_property(collection, query, item_name)
    prop_dict = result.choices[0].message.content
    print(prop_dict)

    sparql_classifier = SPARQLQueryClassifier(api_key = openai.api_key)
    classification = sparql_classifier.classify_query(query)
    print(classification.choices[0].message.content)
    if "ASK WHERE clause" in classification.choices[0].message.content:
        prompt = PromptTemplate(input_variables=["query", "item_name", "item_id", "prop_dict"], template=ask_template)
    elif "SELECT clause" in classification.choices[0].message.content:
        prompt = PromptTemplate(input_variables=["query", "item_name", "item_id", "prop_dict"], template=select_template)

    agent = initialize_sparql_query_agent()
    result = agent.run(prompt.format(query = query, item_name = item_name, item_id = item_id, prop_dict = prop_dict))
    return result
