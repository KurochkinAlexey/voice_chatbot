from langchain_community.utilities import SQLDatabase
from utils import generate_db, get_medical_keywords
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.chains import ConversationChain
from langchain.chains.router import LLMRouterChain, MultiRouteChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser
from langchain.schema import OutputParserException
import json

class TextRouterParser(RouterOutputParser):
    def parse(self, text: str) -> dict:

        text = text.strip()
        cleaned = text.strip('`').replace('json\n', '')
        decoded = cleaned.encode().decode('unicode_escape')
        data = json.loads(decoded)
        
        result = {}
        result['destination'] = data['destination']
        result['next_inputs'] = {'input': data['query']}
        return result

class LLMChat:
    def __init__(self, db_name='medical.db', llm_type='qwen2.5:7b', sql_verbose=True, sql_tool_type='openai-tools'):
        
        db = self.init_db(db_name)
        llm = ChatOllama(model=llm_type)
        
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_agent = create_sql_agent(llm, sql_toolkit, agent_type=sql_tool_type, verbose=sql_verbose)
        general_prompt = PromptTemplate(
        template="""You are a helpful, friendly assistant for general conversation. \
                Respond to the user in a natural, conversational tone. \
                Do NOT discuss medical topics or generate SQL. \
                If the user asks about health, medicine, or data, politely decline and suggest asking a medical professional.

                **Examples of Behavior:**
                - User: "What's the capital of France?" → Answer normally.
                - User: "How do I treat a headache?" → Refuse and redirect to a doctor.

                Current Conversation:
                {history}
                User: {input}
                Assistant:""",
                    input_variables=["input", "history"]
                )

        general_chain = ConversationChain(
            llm=llm,
            prompt=general_prompt,
            memory=ConversationBufferMemory()  # Optional but useful for chat history
        )
        
        destinations = ["medical", "general"]

        router_template = """Given the user query, route it to either the 'medical' or 'general' chain.
                            
                            Examples:
                            - "How many surgeons are interested in cardiology?" -> medical
                            - "How's the weather today?" -> general
                            - "How many doctors work in public institutions?" -> medical
                            - "Tell me a joke" -> general
                            - "How many doctors with name John we have in our database" -> medical

                            In addition, if there are words similar to {} """.format(', '.join(get_medical_keywords())) +\
                            """in query, you should route it to 'medical' chain

                            Query: {input}
                            
                            Output should be in form of dict 'destination': 'general'/'medical' and 'query': {input}
"""
        


        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=TextRouterParser(),  # Use our custom parser
        )
        
        router_chain = LLMRouterChain.from_llm(
                            llm=llm,
                            prompt=router_prompt,
                        )
        
        route_chains = {
            "medical": sql_agent,
            "general": general_chain
        }

        self.multi_route_chain = MultiRouteChain(
            router_chain=router_chain,
            destination_chains=route_chains,
            default_chain=general_chain  # Fallback to general chat
        )
        

    def init_db(self, db_name):
        
        db = SQLDatabase.from_uri("sqlite:///{}".format(db_name))
        if not db.get_table_names():
            generate_db(db_name=db_name)
            db = SQLDatabase.from_uri("sqlite:///{}".format(db_name))
        
        return db
    
    def respond(self, query):
        result = self.multi_route_chain.invoke(query)
        if 'response' not in result.keys():
            result['response'] = result['output']
        return result