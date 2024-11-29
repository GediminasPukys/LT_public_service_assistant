import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from typing import List, Dict, Tuple
import json
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import weaviate.classes as wvc

wcd_url = st.secrets["weaviate_credentials"]["url"]
weaviate_token = st.secrets["weaviate_credentials"]["token"]
openai_api_key = st.secrets["openai_credentials"]["api_key"]


class PublicServiceCategories:
    CATEGORIES = {
        # Identity & Immigration Services
        'IDENTITY_DOCUMENTS': 'Identity Document Services (Passports, ID cards, Personal documents)',
        'RESIDENCE_PERMITS': 'Residence Permit Services (Temporary, Permanent residence)',
        'CITIZENSHIP_SERVICES': 'Citizenship Application and Processing',
        'VISA_SERVICES': 'Visa Application and Processing',
        'IMMIGRATION_SUPPORT': 'Immigration Support Services',
        'FOREIGN_REGISTRATION': 'Foreign National Registration Services',

        # Environment & Waste Management
        'ENVIRONMENTAL_SERVICES': 'Environmental Protection and Management',
        'WASTE_MANAGEMENT': 'Waste Collection and Management Services',
        'ENVIRONMENTAL_PERMITS': 'Environmental Permits and Licensing',
        'RECYCLING_SERVICES': 'Recycling and Waste Recovery Services',

        # Civil Registry Services
        'BIRTH_REGISTRATION': 'Birth Registration and Certificates',
        'DEATH_REGISTRATION': 'Death Registration and Certificates',
        'MARRIAGE_REGISTRATION': 'Marriage Registration and Certificates',
        'CIVIL_STATUS': 'Civil Status Changes and Documentation',
        'ADDRESS_REGISTRATION': 'Address Registration and Declaration',

        # Health & Wellbeing
        'HEALTH_MEDICAL': 'Healthcare and Medical Services',
        'MENTAL_HEALTH': 'Mental Health Services',
        'PUBLIC_HEALTH': 'Public Health and Prevention',
        'ELDERLY_CARE': 'Elderly Care Services',
        'DISABILITY_SERVICES': 'Disability Support Services',
        'EMERGENCY_MEDICAL': 'Emergency Medical Services',

        # Social Services & Welfare
        'SOCIAL_ASSISTANCE': 'Social Assistance and Support',
        'CHILD_PROTECTION': 'Child Protection Services',
        'FAMILY_SERVICES': 'Family Support Services',
        'UNEMPLOYMENT_BENEFITS': 'Unemployment Benefits',
        'PENSION_SERVICES': 'Pension and Retirement Services',
        'SOCIAL_HOUSING': 'Social Housing Services',
        'YOUTH_SERVICES': 'Youth Support Services',

        # Education & Training
        'PRIMARY_EDUCATION': 'Primary Education',
        'SECONDARY_EDUCATION': 'Secondary Education',
        'HIGHER_EDUCATION': 'Higher Education',
        'VOCATIONAL_TRAINING': 'Vocational Training',
        'ADULT_EDUCATION': 'Adult Education',
        'SPECIAL_EDUCATION': 'Special Education',
        'EDUCATIONAL_SUPPORT': 'Educational Support Services',

        # Employment & Labor
        'JOB_SEARCH': 'Job Search and Employment Services',
        'CAREER_GUIDANCE': 'Career Guidance',
        'LABOR_RIGHTS': 'Labor Rights and Protection',
        'WORK_SAFETY': 'Workplace Safety and Health',
        'PROFESSIONAL_CERTIFICATION': 'Professional Certification',

        # Justice & Legal
        'COURT_SERVICES': 'Court Services',
        'LEGAL_AID': 'Legal Aid',
        'VICTIM_SUPPORT': 'Victim Support Services',
        'CRIMINAL_JUSTICE': 'Criminal Justice Services',
        'CIVIL_JUSTICE': 'Civil Justice Services',
        'DISPUTE_RESOLUTION': 'Dispute Resolution',
        'NOTARY_SERVICES': 'Notary Services',

        # Transportation & Infrastructure
        'PUBLIC_TRANSPORT': 'Public Transportation',
        'ROAD_SERVICES': 'Road and Highway Services',
        'VEHICLE_REGISTRATION': 'Vehicle Registration',
        'DRIVER_LICENSING': 'Driver Licensing',
        'TRAFFIC_MANAGEMENT': 'Traffic Management',

        # Urban Planning & Development
        'URBAN_PLANNING': 'Urban Planning and Development Services',
        'CONSTRUCTION_PERMITS': 'Construction Permits and Approvals',
        'LAND_MANAGEMENT': 'Land Use and Management Services',
        'PROPERTY_REGISTRATION': 'Property Registration and Records',

        # Business & Enterprise
        'BUSINESS_REGISTRATION': 'Business Registration and Licensing',
        'BUSINESS_PERMITS': 'Business Permits and Certificates',
        'TRADE_SERVICES': 'Import/Export and Trade Services',
        'BUSINESS_REPORTING': 'Business Reporting and Compliance',

        # Communication & Information
        'POSTAL_SERVICES': 'Postal Services',
        'TELECOMMUNICATIONS': 'Telecommunications',
        'PUBLIC_INFORMATION': 'Public Information Services',
        'DIGITAL_SERVICES': 'Digital Services',
        'MEDIA_SERVICES': 'Media Services',

        # Data & Statistics
        'PUBLIC_REGISTERS': 'Public Register Services',
        'DATA_ACCESS': 'Data Access and Information Requests',
        'STATISTICAL_SERVICES': 'Statistical Information Services',
        'RESEARCH_SERVICES': 'Research and Analysis Services'

    }


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )


def setup_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key(weaviate_token),
        headers={"X-OpenAI-Api-Key": openai_api_key}
    )


def identify_user_intent_categories(query: str, llm: ChatOpenAI) -> List[str]:
    """Identify relevant service categories based on user query"""
    template = """Analyze the following user query and identify the most relevant service categories.
    Use only the category codes from the following list:

    {categories}

    User Query: {query}

    Rules:
    1. Return only the category CODES (e.g., HEALTH_MEDICAL, SOCIAL_ASSISTANCE)
    2. Return up to 3 most relevant categories
    3. Return the codes as a comma-separated list
    4. Be specific and accurate in category assignment

    Return ONLY the category codes, nothing else"""

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = CommaSeparatedListOutputParser()

    categories_str = "\n".join([f"- {code}: {desc}" for code, desc in PublicServiceCategories.CATEGORIES.items()])

    chain = prompt | llm | output_parser
    result = chain.invoke({"categories": categories_str, "query": query})

    return result

#
# def get_services_by_categories(client, categories: List[str], query: str, limit: int = 10) -> List[Dict]:
#     """Perform hybrid search for services based on categories and query"""
#     try:
#         collection = client.collections.get("ServiceDescription")
#
#         # Create filter for ENRICHED_labels using ContainsAny
#         category_filter = wvc.query.Filter.by_property("eNRICHED_labels").contains_any(categories)
#
#         # Execute hybrid search with filter
#         response = collection.query.hybrid(
#             query=query,
#             alpha=0.5,
#             filters=category_filter,
#             return_metadata=wvc.query.MetadataQuery(
#                 score=True,
#                 explain_score=True
#             ),
#             limit=limit
#         )
#
#         # Process and return results
#         results = []
#         for obj in response.objects:
#             result = {
#                 "properties": obj.properties,
#                 "score": obj.metadata.score,
#                 "explain_score": obj.metadata.explain_score
#             }
#             results.append(result)
#
#             # Debug print
#             print(f"Score: {obj.metadata.score}")
#             print(f"Properties: {obj.properties}")
#             print(f"Explanation: {obj.metadata.explain_score}")
#             print("---")
#
#         return results
#
#     except Exception as e:
#         print(f"Error in hybrid search: {str(e)}")
#         return []
#


def get_services_by_categories(client, categories: List[str], query: str, limit: int = 10) -> List[Dict]:
    """Perform hybrid search for services based on categories and query"""
    try:
        collection = client.collections.get("ServiceDescription")
        query_with_categories = query + ('  '.join(categories))

        # Create filter for ENRICHED_labels using ContainsAny
        # category_filter = wvc.query.Filter.by_property("eNRICHED_labels").contains_any(categories)

        # Execute hybrid search with filter
        response = collection.query.hybrid(
            query=query_with_categories,
            alpha=0.75,
            # filters=category_filter,
            return_metadata=wvc.query.MetadataQuery(
                score=True,
                explain_score=True
            ),
            limit=limit
        )

        with st.expander("🔍 Debug Information", expanded=False):
            i = 1
            for obj in response.objects:
                st.write(f'Iteration # : {i}')
                st.write(f'score : {obj.metadata.score}')
                st.write(
                    {
                        'cATEGORIES': obj.properties['cATEGORIES'],
                        'sHORT_DESCRIPTION': obj.properties['sHORT_DESCRIPTION'],
                        'full_description': obj.properties['full_description'],
                        'eNRICHED_labels': obj.properties['eNRICHED_labels'],
                        'sHORT_NAME': obj.properties['sHORT_NAME'],
                        'lIFE_EVENTS': obj.properties['lIFE_EVENTS'],
                        'pOPULARITY': obj.properties['pOPULARITY'],
                        'lONG_NAME': obj.properties['lONG_NAME'],
                        'kEYWORDS': obj.properties['kEYWORDS'],
                        'iD': obj.properties['iD'],
                        'cOMBINED_NAME': obj.properties['cOMBINED_NAME'],
                        'pROVIDER_NAMES': obj.properties['pROVIDER_NAMES'],
                    })
                i += 1

        # Process and return results
        results = []

        for obj in response.objects:
            result = {
                "properties": obj.properties,
                "score": obj.metadata.score,
                "explain_score": obj.metadata.explain_score
            }
            results.append(result)

            # Debug print
            print(f"Score: {obj.metadata.score}")
            print(f"Properties: {obj.properties}")
            print(f"Explanation: {obj.metadata.explain_score}")
            print("---")

        return results

    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []



from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator, field_validator
from decimal import Decimal


class ServiceEvaluation(BaseModel):
    """Evaluation of a single service's relevance to user query"""

    service_number: int = Field(
        description="Number of the service from the provided list (1-based)",
        gt=0
    )

    service_provider: str = Field(
        description="Name of the service provider"
    )

    relevance_score: float = Field(
        description="How well the service matches the user's needs, from 0 to 1",
        ge=0,
        le=1
    )

    relevance_explanation: str = Field(
        description="Detailed explanation of why this service matches or doesn't match the request"
    )

    action_steps: List[str] = Field(
        description="List of concrete steps the user should take to use this service",
        min_length=1
    )

    @field_validator('relevance_score')
    def validate_score(cls, v):
        return round(float(v), 2)


class ServiceEvaluations(BaseModel):
    """Collection of evaluated services with additional context"""

    evaluations: List[ServiceEvaluation] = Field(
        description="List of evaluated services",
        max_length=3
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="List of information that would help provide better results"
    )

    suggested_questions: Optional[List[str]] = Field(
        default=None,
        description="Follow-up questions that might help user get better results"
    )


def evaluate_and_format_services(prompt: str, services: List[Dict], llm: ChatOpenAI) -> str:
    """
    Evaluates services and returns top matching ones with detailed information.
    """
    # Format services for evaluation
    services_list = "\n\n".join([
        f"Service #{i + 1}:\n"
        f"Name: {service['properties'].get('sHORT_NAME', '')}\n"
        f"Description: {service['properties'].get('full_description', '')}\n"
        f"Popularity: {service['properties'].get('pOPULARITY', '')}\n"
        f"Provider: {service['properties'].get('pROVIDER_NAMES', '')}\n"
        f"Categories: {service['properties'].get('cATEGORIES', '')}"
        for i, service in enumerate(services)
    ])

    # Create evaluation prompt
    evaluation_prompt = f"""
    Evaluate which services best match the user's query. 
    Evaluate if user intent is for more generic or specific as some services can be provided online, national wide and some in particular locations. 
    More generic intents should consider popularity.
    Compare the services and explain your decisions.
    Consider what information might be missing and what follow-up questions could help.

    User query: {prompt}

    Available services:
    {services_list}

    Provide a structured evaluation including:
    1. Top matching services (max 3)
    2. For each service:
       - Relevance score and explanation
       - Concrete steps for using the service
    3. Any missing information that would help provide better results
    4. Suggested follow-up questions

    Be specific and practical in your evaluation.
    """

    try:
        # Create structured output model
        structured_llm = llm.with_structured_output(ServiceEvaluations)

        with st.expander("🔍 Evaluation prompt", expanded=False):
            st.write(evaluation_prompt)
        # Get evaluations
        evaluations = structured_llm.invoke(evaluation_prompt)

        # Format response
        result = "🎯 Tinkamiausios paslaugos jūsų užklausai:\n\n"

        for eval in evaluations.evaluations:
            service_idx = eval.service_number - 1
            if service_idx >= 0 and service_idx < len(services):
                service = services[service_idx]['properties']

                result += f"""
                🔷 {service.get('sHORT_NAME', '')}

                📋 Aprašymas: {service.get('SHORT_DESCRIPTION', '')}

                ✨ Kodėl ši paslauga tinka:
                {eval.relevance_explanation}

                ⭐ Atitikimo balas: {eval.relevance_score:.2f}

                🏢 Paslaugos teikėjas: {eval.service_provider}

                -------------------
                """

        # Add missing information and follow-up questions if available
        if evaluations.missing_information:
            result += "\n📝 Papildoma informacija, kuri padėtų tiksliau atsakyti:\n"
            result += "\n".join(f"• {info}" for info in evaluations.missing_information)

        if evaluations.suggested_questions:
            result += "\n\n❓ Galbūt norėtumėte sužinoti:\n"
            result += "\n".join(f"• {q}" for q in evaluations.suggested_questions)

        return result

    except Exception as e:
        print(f"Error in service evaluation: {str(e)}")
        return "Atsiprašome, įvyko klaida vertinant paslaugas. Bandykite dar kartą."
def format_service_results(evaluated_services: List[Dict]) -> str:
    """Format the evaluated services for display"""
    if not evaluated_services:
        return "Atsiprašau, bet neradu tinkamų paslaugų. Pabandykite perfrazuoti užklausą."

    response = "🎯 Radau šias tinkamiausias paslaugas:\n\n"

    for service in evaluated_services[:5]:  # Show top 5 results
        service_data = service['service_data']
        response += f"""
        🔷 **{service_data.get('COMBINED_NAME', '')}**

        Atitikimas: {service['relevance_score']:.2f}

        📝 {service_data.get('SHORT_DESCRIPTION', '')}

        ✨ Kodėl ši paslauga tinka:
        {service['explanation']}

        🎯 Pagrindiniai aspektai:
        {' • '.join(service['highlights'])}

        🏢 Paslaugos teikėjai:
        {service_data.get('PROVIDER_NAMES', '')}

        -------------------
        """

    return response

import json


def translate_query(query: str, llm: ChatOpenAI) -> str:
    """Translate user query to Lithuanian"""
    translation_prompt = f"""
    Translate the following text to Lithuanian. Return ONLY the translation, nothing else:
    {query}
    """

    try:
        translated_query = llm.predict(translation_prompt)
        return translated_query.strip()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return query  # Fallback to original query if translation fails


def process_user_query(prompt: str, client, llm: ChatOpenAI) -> Tuple[str, List[Dict]]:
    """Process user query and return relevant services"""
    # Step 1: Identify relevant categories based on user intent
    intent_categories = identify_user_intent_categories(prompt, llm)
    st.write("Identified categories:", intent_categories)  # Debug print

    # Step 2: Translate query to Lithuanian for better vector search
    translated_prompt = translate_query(prompt, llm)
    st.write("Translated query:", translated_prompt)  # Debug print

    # Step 3: Get services using hybrid search and category filtering
    services = get_services_by_categories(client, intent_categories, translated_prompt)
    st.write(f"Found {len(services)} services")  # Debug print

    # Step 4: Evaluate and format results
    response = evaluate_and_format_services(prompt, services, llm)

    return response, services

def main():
    st.title("🇱🇹 Valstybinių paslaugų asistentas")

    init_session_state()

    try:
        client = setup_weaviate_client()
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_api_key)
    except Exception as e:
        st.error("Klaida jungiantis prie paslaugų duomenų bazės")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Kokios paslaugos ieškote?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and display results
        with st.chat_message("assistant"):
            try:
                with st.spinner("Ieškau tinkamiausių paslaugų..."):
                    st.write(prompt)
                    response, evaluated_services = process_user_query(prompt, client, llm)

                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

                    # Display debug information in sidebar if needed
                    with st.sidebar:
                        if st.checkbox("Rodyti detalią informaciją"):
                            st.json([{
                                "id": s['service_data'].get('iD'),
                                "name": s['service_data'].get('COMBINED_NAME'),
                                "score": s['relevance_score'],
                                "categories": s['service_data'].get('ENRICHED_labels', [])
                            } for s in evaluated_services])

            except Exception as e:
                st.error("Įvyko klaida apdorojant užklausą")
                print(f"Error: {str(e)}")

    # Clear chat button
    with st.sidebar:
        if st.button("Išvalyti pokalbį"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.rerun()


if __name__ == "__main__":
    main()