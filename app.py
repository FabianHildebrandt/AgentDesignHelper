import gradio as gr
import yaml
import os
import json
from pathlib import Path
from llm import LLMProvider
from typing import Optional
import tiktoken
from typing import Dict
from structured_outputs import AgentGraph, NodeBase, Edge, AgentDescriptionEval, ConcisenessEval,CompletenessEval, ConsistencyEval
import networkx as nx
from pyvis.network import Network



def build_networkx_graph(agent_graph: AgentGraph) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add root Orchestrator node
    orchestrator_node = NodeBase(
        canonical_id="Orchestrator",
        label="Orchestrator",
        type="agent",
        description="The orchestrator agent processes the user query,\n sets up a plan given the available agents and controls the task execution flow (agent calling, result and error handling) \nuntil the final answer is produced."
    )
    G.add_node(orchestrator_node.canonical_id, **orchestrator_node.model_dump())

    # Add all other nodes with metadata
    for node in agent_graph.nodes:
        G.add_node(node.canonical_id, **node.model_dump())

    # Add all edges from AgentGraph
    for edge in agent_graph.edges:
        G.add_edge(edge.source, edge.target, relation=edge.relation, description=edge.description)

    # Connect root-level agent nodes to Orchestrator
    for node in agent_graph.nodes:
        has_orchestrator = False
        # check each node whether it has a supervisor node
        for source, target, metadata in list(G.in_edges(node.canonical_id, data=True)):
            if "manages" in metadata.get("relation", ""):
                has_orchestrator = True
                break
        if node.type == "agent" and not has_orchestrator:
            G.add_edge("Orchestrator", node.canonical_id, relation="manages")

    return G

def store_graph_visualization(graph : Dict, target_path : str = "agent_responsibility_graph.html", graph_style : Dict = None):
    if graph_style == None:
        graph_style = {
            "bgcolor": "#ffffff",         # white background
            "font_color": "#000000",      # black text
            "node_color": "#97C2FC",      # default PyVis blue
            "agent_color": "#FFF2CC",      # default PyVis blue
            "capability_color": "#DAE8FC",      # default PyVis blue
            "task_color": "#D5E8D4",      # default PyVis blue
            "edge_color": "#848484",       # default PyVis gray
            "redundancy_color" : "#F70000"
        }
    # Initialize PyVis network with remote CDN for Gradio compatibility
    net = Network(
        notebook=True,  # Set to False for Gradio HTML component
        bgcolor=graph_style['bgcolor'],
        font_color=graph_style["font_color"],
        directed=graph.is_directed(), 
        height="100vh",  # Fixed height instead of 100vh
        width="100%",
        cdn_resources='remote'  # Use remote CDN instead of local
    )

    # Add nodes with label inside and tooltip with metadata
    for node_id, data in graph.nodes(data=True):
        label = data.get("label", node_id)
        node_type = data.get("type", "unknown")
        description = data.get("description", "")
        
        tooltip = f"Canonical ID: {node_id}\n" \
                  f"Type: {node_type}\n" \
                  f"Description: {description}"
        
        if node_type == "agent":
            node_color = graph_style["agent_color"]
        elif node_type == "capability":
            node_color = graph_style["capability_color"]
        elif node_type == "task":
            node_color = graph_style["task_color"]
        else:
            node_color = graph_style["node_color"]

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=node_color,
            shape="box",
            borderWidth=2,
            margin=10,
            font={"size": 18, "align": "center"},
            widthConstraint={"minimum": 100}
        )

    # Add edges with labels and hover tooltips
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation", "related_to")
        description = data.get("description", "N/A")
        tooltip = f"Relation: {relation}\n"\
                f"Description: {description}"
        
        if "redundant" not in relation and "redundant" not in description:
            color = graph_style["edge_color"]
        else:
            color = graph_style["redundancy_color"]
        
        net.add_edge(
            source,
            target,
            label=relation,
            title=tooltip,
            color=color,
            arrows="to",
            font={"align": "top", "size": 14}
        )

    # Custom physics and layout options
    options = json.dumps({
        "physics": {
            "forceAtlas2Based": {
                "springLength": 100,
                "gravitationalConstant": -150
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "layout": {
            "improvedLayout": True
        },
        "nodes": {
            "shape": "box",
            "shapeProperties": {
                "borderRadius": 10
            }
        }
    })

    net.set_options(options)
    net.save_graph(target_path)
    # For Gradio, we need to generate HTML content directly without saving to file
    html_content = net.generate_html()
    return html_content

def test_graph_visualization():
    """Create a simple test graph to verify visualization works"""
    graph = AgentGraph(
        name="Test Agent Graph",
        description="This is a test graph with two agents",
        nodes=[
            NodeBase(
                canonical_id="Agent1",
                label="Agent1",
                type="agent",
                description="The first test agent"
            ),
            NodeBase(
                canonical_id="Agent2",
                label="Agent2",
                type="agent",
                description="The second test agent"
            )
        ],
        edges=[
            Edge(
                source="Agent1",
                target="Agent2",
                relation="manages",
                description="Agent 1 manages Agent 2."
            )
        ]
    )
    
    test_graph = build_networkx_graph(graph)
    styleguide = None
    # try to read the styleguide
    if os.path.exists("styleguide.yaml"):
        with open("styleguide.yaml", "r", encoding="utf-8") as f:
            styleguide = yaml.safe_load(f)
    try:
        html_content = store_graph_visualization(test_graph, graph_style=styleguide)
        if html_content:  # Basic validation
            return "✅ Graph visualization test passed. Stored the agent responsibility graph as `agent_responsibility_graph.html`."
        else:
            return "❌ Graph visualization test failed - empty content"
    except Exception as e:
        gr.Warning(f"❌ Graph visualization test failed: {str(e)}")
        return (f"❌ Graph visualization test failed: {str(e)}")

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error finding config.yaml file. Please make sure that there is a config.yaml placed in the same directory.")
        # Return default config if file not found
        return {"llm": []}
    
def load_prompts():
    """Load prompts from prompts.yaml"""
    try:
        with open('prompts.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Error finding prompts.yaml file. Please make sure that there is a prompts.yaml placed in the same directory.")
    except Exception as e:
        raise Exception(f"Error reading prompts.yaml file: {str(e)}")

def load_template():
    """Load template agent description from example.md"""
    try: 
        with open("example.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: example.md file not found. Using empty template.")
        return ""
    except Exception as e:
        raise Exception(f"Error reading example.md file: {str(e)}")

def get_description_template():
    """Get the description template from prompts.yaml"""
    try:
        prompts = load_prompts()
        return prompts.get("description_template", "")
    except:
        return "Error loading description template from prompts.yaml"

def load_wiki_content():
    """Load the agent description writing wiki content"""
    try:
        with open("writing_agent_descriptions.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Writing Effective Agent Descriptions\n\nWiki content not found. Please ensure writing_agent_descriptions.md exists in the project directory."
    except Exception as e:
        return f"# Error Loading Wiki\n\nError reading wiki content: {str(e)}"


def get_llm_defaults():
    """Get default LLM configurations for each provider type"""
    config = load_config()
    llm_configs = config.get("llm", [])
    
    defaults = {}
    for llm_config in llm_configs:
        provider_type = llm_config.get("type", "")
        defaults[provider_type] = llm_config
    
    return defaults

def update_llm_fields(provider_type):
    """Update LLM configuration fields based on selected provider"""
    defaults = get_llm_defaults()
    
    if provider_type not in defaults:
        return "", "", "", 0.0, ""
    
    config = defaults[provider_type]
    
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    temperature = config.get("temperature", 0.0)
    api_version = config.get("api_version", "")
    
    return api_key, base_url, model, temperature, api_version

def optimize_description(agent_description, llm_config):
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before optimizing the agent description.")
    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the description optimization.")
    
    llm = LLMProvider(config=llm_config)
    system_prompt = "\n\n".join([
        prompts["description_creation_prompt"],
        "You need to follow the provided template:",
        prompts["description_template"],
        "This is an example of a long agent summary description:",
        prompts["long_description_example"],
        "This is an example of the shortened optimal description:",
        prompts["good_description_example"]
    ])

    user_prompt = "\n\n".join([
            "This is the provided user input:",
            agent_description,
            "Please create concise, complete and consistent agent description(s) as described in your system instructions."
        ])

    optimized_description = llm.call_llm(system_prompt, user_prompt).content    

    return optimized_description

def reduce_overlap(agent_description, llm_config):
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before optimizing the agent description.")
    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the description optimization.")
    
    llm = LLMProvider(config=llm_config)
    system_prompt = "\n\n".join([
        prompts["description_creation_prompt"],
        "You need to follow the provided template:",
        prompts["description_template"],
        "This is an example of an optimal description:",
        prompts["good_description_example"],
        "You are given another task, that is really important:",
        prompts["description_overlap_reduction"],
        "This is an example case, that should show you how to reduce the overlap:",
        prompts["bad_description_examples"],
        "This is the overlap analysis:",
        prompts["overlap_analysis_example"],
        prompts["overlap_reduction_example"]
    ])

    user_prompt = "\n\n".join([
            "This is the provided user input:",
            agent_description,
            "Please follow the instructions to reduce the overlap and create concise, complete and consistent agent description(s) as described in your system instructions."
        ])

    optimized_description = llm.call_llm(system_prompt, user_prompt).content    

    return optimized_description

def parse_directory_for_agents(directory_path):
    """Parse a directory for JSON, YAML and text files containing agent descriptions"""
    if not directory_path or not os.path.exists(directory_path):
        return "Error: Please select a valid directory path."
    
    agent_descriptions = []
    directory = Path(directory_path)
    
    # Look for JSON and text files
    for file_path in directory.glob("*"):
        description_str = ""
        data = None
        if file_path.suffix.lower() in ['.json', '.txt', '.yml', '.yaml']:
            try:
                if file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        description_str = "\n".join([f'{key}: {value}' for key,value in data.items()])
                elif file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        description_str = "\n".join([f'{key}: {value}' for key,value in data.items()])
                elif file_path.suffix.lower() == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        description_str = f.read().strip()
                
                # Add to agent descriptions if we have content
                if description_str.strip():
                    agent_descriptions.append(f'Agent from {file_path.name}:\n{description_str}')
                            
            except Exception as e:
                agent_descriptions.append(f"Error reading {file_path.name}: {str(e)}\n")
    
    if not agent_descriptions:
        return "No agent descriptions found in the selected directory."
    
    return "\n---\n\n".join(agent_descriptions)

def analyze_overlap(agent_descriptions_text, llm_config):
    """Analyze overlap between multiple agent descriptions (placeholder function)"""
    if not agent_descriptions_text.strip():
        return "Please provide agent descriptions to analyze."
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before performing overlap analysis.")
    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the overlap analysis.")
    
    llm = LLMProvider(config=llm_config)
    system_prompt = "\n\n".join([
        prompts["overlap_analysis"],
        "Here you can find some exemplary agent descriptions and the corresponding ambuity report. This should help you to understand the task:",
        "[EXAMPLES]",
        prompts["bad_description_examples"],
        prompts["good_description_example"],
        "[END EXAMPLES]",
        prompts["overlap_analysis_example"],
    ])

    user_prompt = "\n\n".join([
            "This is the agent description:",
            agent_descriptions_text,
            "Please create an overlap analysis as described in your system instructions."
        ])

    optimized_description = llm.call_llm(system_prompt, user_prompt).content    

    return optimized_description

def create_agent_responsibility_graph(agent_descriptions_text, llm_config, progress = gr.Progress()):
    """Create a graphical visualization of the Multi-Agent System containing agents, capabilities and tasks with their relationships."""
    if not agent_descriptions_text.strip():

        return "### Please provide agent descriptions to analyze!"
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before performing dynamic overlap analysis.")

    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the dynamic overlap analysis.")
    progress(0, desc="Starting static overlap analysis...")
    static_overlap_analysis = analyze_overlap(agent_descriptions_text, llm_config)
    
    progress(0.33, desc="Static overlap analysis completed. Starting the extraction of graph nodes and edges...")
    llm = LLMProvider(config=llm_config)

    # Entity extraction step
    system_prompt = "\n\n".join([
        prompts["agent_graph_prompt"],
        "This example demonstrates how to create the agent resposibility graph based on agent descriptions.",
        prompts["bad_description_examples"],
        prompts["good_description_example"],
        prompts["overlap_analysis_example"],
        prompts["agent_graph_example"]
    ])

    user_prompt_text = "\n\n".join([
        "This is the overview of the agent descriptions",
        agent_descriptions_text,
        "This is the overlap analysis:",
        static_overlap_analysis,
        "Create the agent graph based on the descriptions."
    ])

    agent_responsibility_graph = llm.call_llm(system_prompt, user_prompt_text, AgentGraph) 
    progress(0.66, desc="Extraction of the graph nodes and edges completed. Creating and rendering the graph...")
    styleguide = None
    # try to read the styleguide
    if os.path.exists("styleguide.yaml"):
        with open("styleguide.yaml", "r", encoding="utf-8") as f:
            styleguide = yaml.safe_load(f)

    try: 
        nx_graph = build_networkx_graph(agent_responsibility_graph)
        target_path="agent_responsibility_graph.html"
        html_graph = store_graph_visualization(graph = nx_graph, graph_style=styleguide)
        if html_graph:
            message = f"✅ Agent responsibility graph created successfully! The visualization can be found in {target_path}."
        else:
            message = "❌ Agent responsibility graph creation failed - empty content."
        return message
    except Exception as e:
        gr.Warning(f"An error occurred: {str(e)}!", duration=5)
        return f"Error creating visualization: {str(e)}"


def analyze_dynamic_overlap(agent_descriptions_text, planning_prompt, user_query, llm_config, progress = gr.Progress()):
    """Analyze dynamic overlap between agents for a specific user query and planning system"""
    if not agent_descriptions_text.strip():
        return "Please provide agent descriptions to analyze."
    if not user_query.strip():
        return "Please provide a user query to analyze potential conflicts."
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before performing dynamic overlap analysis.")

    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the dynamic overlap analysis.")
    
    progress(0, desc="Starting static overlap analysis...")
    static_overlap_analysis = analyze_overlap(agent_descriptions_text, llm_config)
    

    progress(0.5, desc="Static overlap analysis completed.")

    progress(0.51, desc="Starting dynamic overlap analysis...")
    llm = LLMProvider(config=llm_config)

    system_prompt = "\n\n".join([
        prompts.get("dynamic_overlap_analysis", "Analyze potential conflicts between agents during orchestration for a specific user query."),
        "This is the orchestrator system prompt",
        planning_prompt if planning_prompt.strip() else "No specific planning system prompt provided.",
        "This is the overview of the agent descriptions",
        agent_descriptions_text,
        "These are the results of the static overlap analysis:",
        static_overlap_analysis
    ])

    user_prompt_text = "\n\n".join([
        "User Query to analyze:",
        user_query,
    ])

    analysis_result = llm.call_llm(system_prompt, user_prompt_text).content    

    return analysis_result


def evaluate_description(agent_description, llm_config, max_tokens=200, description_template=""):
    """Evaluate a single agent description using LLM and token count"""
    if not agent_description.strip():
        return "Please provide agent descriptions to analyze.", {}, {}, {}
    if not llm_config or not llm_config.get("api_key", "").strip():
        raise ValueError("Please make sure to configure the LLM endpoint and API key before evaluating the agent description.")
    
    # Calculate token count using tiktoken
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        token_count = len(encoding.encode(agent_description))
        token_count_ok = token_count <= max_tokens
    except Exception as e:
        token_count = 0
        token_count_ok = False
        print(f"Error calculating token count: {e}")
    
    try:
        prompts = load_prompts()
    except:
        raise ValueError("Couldn't find any prompts related to the description evaluation.")
    
    llm = LLMProvider(config=llm_config)
    system_prompt = "\n\n".join([
        prompts["description_evaluation"],
        "This template should be used to evaluate the consistency and adherence to the styleguide:",
        description_template or prompts["description_template"],
        "Here you can find some exemplary agent descriptions and their evaluations.",
        "[EXAMPLES]",
        prompts["bad_description_examples"],
        prompts["good_description_example"],
        "[END EXAMPLES]",
        prompts["description_evaluation_examples"],
        "Remember to strictly follow the provided structured output format."
    ])

    user_prompt = "\n\n".join([
            "This is the agent description:",
            agent_description,
            "Please evaluate the agent description as described in your system instructions."
        ])

    try:
        response = llm.call_llm(system_prompt, user_prompt, AgentDescriptionEval)
        
        # Update token count in the response
        if hasattr(response, 'conciseness'):
            response.conciseness.token_count = token_count
            response.conciseness.allowed_token_count = token_count_ok
        
        # Prepare results for UI
        feedback = response.feedback or "No feedback provided."
        
        # Completeness results
        completeness_results = {
            "contains_name": response.completeness.contains_name,
            "contains_role": response.completeness.contains_role,
            "contains_capabilities": response.completeness.contains_capabilities,
            "contains_examples": response.completeness.contains_examples,
            "contains_edgecases": response.completeness.contains_edgecases
        }
        
        # Consistency results
        consistency_results = {
            "has_conform_structure": response.consistency.has_conform_structure,
            "has_conform_name": response.consistency.has_conform_name
        }
        
        # Conciseness results
        conciseness_results = {
            "is_concise": response.conciseness.is_concise,
            "is_relevant": response.conciseness.is_relevant,
            "avoids_redundancy": response.conciseness.avoids_redundancy,
            "token_count": token_count,
            "allowed_token_count": token_count_ok
        }
        
        return feedback, completeness_results, consistency_results, conciseness_results
        
    except Exception as e:
        error_msg = f"Error during evaluation: {str(e)}"
        return error_msg, {}, {}, {}

def get_template_token_count(template_str):

    if type(template_str) == str:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        token_count = len(encoding.encode(template_str))

    return token_count if token_count else 0 

def main():
    # Get available provider types from config
    defaults = get_llm_defaults()
    template_agent_description = load_template()
    provider_choices = list(defaults.keys()) if defaults else ["openai", "googlegenai", "claude", "azureopenai"]
    
    with gr.Blocks() as application:
        # Initialize LLM config state
        llm_config_state = gr.State({
            "temperature": 0.0,
            "model": "",
            "type": provider_choices[0] if provider_choices else "azureopenai",
            "api_key": "",
            "base_url": "",
            "api_version": ""
        })
        
        gr.Markdown("# 🤖 Agent Design Helper")
        with gr.Row():
            with gr.Column(scale=1):  # Image on the far left
                gr.Image(value="resources/agentdesignhelper.png", container=False, show_download_button=False, show_fullscreen_button=False)

            with gr.Column(scale=4):  # Markdown starts next to the image
                gr.Markdown("## The agent design helper supports you to write better agent descriptions.")
                gr.Markdown("""
### Features: 
- Tutorial: How to write good agent descriptions.                            
- Description optimization: The description optimizer supports you to write high-quality agent descriptions for your specialized agent
- Description evaluation: Evaluate your agent description for certain quality criteria.
- Static overlap analysis: Analyze overlap between multiple agent descriptions and get guidance on resolving conflicts. 
- Dynamic overlap analysis: Test your user queries to analyze the overlap/ conflicts between multiple agents that can arise in multi-agent orchestration.
- Agent responsibility graph: Create a nice html visualization of your multi-agent system.

### Configure your LLM endpoint first to use the features. 
""")
        
                with gr.Accordion("LLM endpoint configuration", open=False):
                    # Provider selection dropdown
                    provider_dropdown = gr.Dropdown(
                        choices=provider_choices,
                        value=provider_choices[0] if provider_choices else "azureopenai",
                        label="LLM Provider",
                        info="Select your LLM provider"
                    )
                    
                    # Configuration fields
                    with gr.Row():
                        with gr.Column():
                            api_key_input = gr.Textbox(
                                label="API Key",
                                placeholder="Enter your API key",
                                info="Your API key for the selected provider",
                                interactive=True,
                                type="password"
                            )
                            
                            base_url_input = gr.Textbox(
                                label="Base URL (optional)",
                                placeholder="Enter custom base URL if needed",
                                info="Custom endpoint URL (leave empty for default)",
                                interactive=True
                            )
                        
                        with gr.Column():
                            model_input = gr.Textbox(
                                label="Model",
                                placeholder="Enter model name",
                                info="The specific model to use (e.g., gpt-4o, gemini-2.0-flash-001)",
                                interactive=True
                            )
                            
                            temperature_input = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                label="Temperature",
                                info="Controls randomness in responses (0 = deterministic, 2 = very creative)",
                                interactive=True
                            )
                    
                    # API Version field (mainly for Azure OpenAI)
                    api_version_input = gr.Textbox(
                        label="API Version (Azure OpenAI only)",
                        placeholder="e.g., 2025-01-01-preview",
                        info="Required for Azure OpenAI, leave empty for other providers"
                    )
                    
                    # Function to update LLM config state
                    def update_llm_config_state(provider, api_key, base_url, model, temperature, api_version):
                        return {
                            "type": provider,
                            "api_key": api_key.strip() if api_key else "",
                            "base_url": base_url,
                            "model": model,
                            "temperature": temperature,
                            "api_version": api_version
                        }
                    
                    # Update state when any LLM config field changes
                    for component in [provider_dropdown, api_key_input, base_url_input, model_input, temperature_input, api_version_input]:
                        component.change(
                            fn=update_llm_config_state,
                            inputs=[provider_dropdown, api_key_input, base_url_input, model_input, temperature_input, api_version_input],
                            outputs=[llm_config_state]
                        )
                    
                    # Update fields when provider changes
                    provider_dropdown.change(
                        fn=update_llm_fields,
                        inputs=[provider_dropdown],
                        outputs=[api_key_input, base_url_input, model_input, temperature_input, api_version_input]
                    )
                    
                    # Initialize fields with default values
                    def initialize_llm_config():
                        defaults = update_llm_fields(provider_choices[0] if provider_choices else "azureopenai")
                        return defaults + (update_llm_config_state(
                            provider_choices[0] if provider_choices else "azureopenai",
                            defaults[0], defaults[1], defaults[2], defaults[3], defaults[4]
                        ),)
                    
                    application.load(
                        fn=initialize_llm_config,
                        outputs=[api_key_input, base_url_input, model_input, temperature_input, api_version_input, llm_config_state]
                    )

        with gr.Tabs():
            with gr.Tab("Writing Guide"):
                gr.Markdown("# How to Write Effective Agent Descriptions")
                gr.Markdown("A comprehensive guide for creating high-quality agent descriptions.")
                
                # Load and display the wiki content
                wiki_content = load_wiki_content()
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Image(value="resources/mas_overview.png", container=False, show_download_button=False, show_fullscreen_button=False)
                    with gr.Column(scale=1):
                        gr.Markdown("""
## Key points:
1. Well-crafted agent descriptions are crucial for successful multi-agent systems, as they help understand capabilities, prevent overlap, and set clear expectations.
2. Agent descriptions should include a specific name, clear role description, detailed capabilities, and usage scenarios to showcase the agent's primary use cases.
3. Tools and functions should have concise, descriptive names, with clear documentation of input parameters, returned results, and any limitations or constraints.
4. Identifying and documenting distinctive usage scenarios helps reduce ambiguity and showcases the agent's key functionalities compared to other agents.
5. Recognizing and documenting edge cases, either by design or through testing, helps set realistic expectations and prevents user frustration by addressing the boundaries of the agent's capabilities.

""")
                gr.Markdown(
                    value=wiki_content,
                    elem_classes=["wiki-content"]
                )

                        
            with gr.Tab("Description Optimization"):
                gr.Markdown("## Agent Description Optimizer")
                agent_description = gr.TextArea(lines=10, label="Enter your agent description here.", value=template_agent_description)
                default_input = "## Enter your agent description and optimize it."
                with gr.Row():
                    optimized_description = gr.TextArea(label="Optimized agent description", value=default_input)
                    markdown_view = gr.Markdown(label="Markdown view", value=default_input)
                with gr.Row():
                    optimize_button = gr.Button("Optimize the agent description", variant="primary")
                    reduce_overlap_button = gr.Button("Reduce the overlap")

                optimize_button.click(
                    fn=optimize_description,
                    inputs=[agent_description, llm_config_state],
                    outputs=[optimized_description]
                )
                reduce_overlap_button.click(
                    fn=reduce_overlap,
                    inputs=[agent_description, llm_config_state],
                    outputs=[optimized_description]
                )

                # Update markdown view whenever optimized_description changes
                optimized_description.change(
                    fn=lambda text: text,
                    inputs=[optimized_description],
                    outputs=[markdown_view]
                )
            with gr.Tab("Description Evaluation"):
                gr.Markdown("## Agent Description Evaluation")
                gr.Markdown("Evaluate a single agent description for completeness, consistency, and conciseness.")
                
                # Agent description input
                eval_agent_description = gr.TextArea(
                    lines=10, 
                    label="Enter your agent description here.", 
                    value=template_agent_description.split("Detailed description")[0]
                )
                
                # Configuration accordion (closed by default)
                with gr.Accordion("Evaluation Configuration", open=False):
                    max_tokens_input = gr.Number(
                        label="Maximum Token Count",
                        value=200,
                        minimum=1,
                        maximum=1000,
                        info="Maximum allowed token count for the agent description"
                    )
                    
                    description_template_input = gr.TextArea(
                        label="Description Template",
                        value=get_description_template(),
                        lines=8,
                        info="Template used to evaluate consistency and structure"
                    )

                    template_token_count = gr.Number(
                        label="Template Token Count",
                        info="Token count of the provided template"
                    )
                
                    description_template_input.change(
                        get_template_token_count,
                        inputs=[description_template_input],
                        outputs=[template_token_count]
                    )
                
                # Initialize template token count on app load
                application.load(
                    fn=lambda: get_template_token_count(get_description_template()),
                    outputs=[template_token_count]
                )
                # Evaluate button
                evaluate_button = gr.Button("Evaluate Description", variant="primary")
                
                # Results section
                gr.Markdown("### Evaluation Results")
                
                # Feedback section
                feedback_output = gr.Markdown(
                    label="Feedback",
                    value="**Click 'Evaluate Description' to see the evaluation feedback here.**"
                )
                
                # Evaluation criteria sections
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Completeness Evaluation")
                        gr.Markdown("*Does the description cover all key content types?*")
                        
                        completeness_name = gr.Checkbox(
                            label="Contains Name",
                            info="Description text contains the agent's name",
                            interactive=False
                        )
                        completeness_role = gr.Checkbox(
                            label="Contains Role",
                            info="Description contains a role description",
                            interactive=False
                        )
                        completeness_capabilities = gr.Checkbox(
                            label="Contains Capabilities",
                            info="Description contains agent's capabilities",
                            interactive=False
                        )
                        completeness_examples = gr.Checkbox(
                            label="Contains Examples",
                            info="Description contains specific usage scenarios",
                            interactive=False
                        )
                        completeness_edgecases = gr.Checkbox(
                            label="Contains Edge Cases",
                            info="Description contains limitations/edge cases",
                            interactive=False
                        )
                with gr.Row():    
                    with gr.Column():
                        gr.Markdown("#### Consistency Evaluation")
                        gr.Markdown("*Does the description follow expected structure and naming conventions?*")
                        
                        consistency_structure = gr.Checkbox(
                            label="Conforms to Structure",
                            info="Follows the provided template structure",
                            interactive=False
                        )
                        consistency_name = gr.Checkbox(
                            label="Conforms to Naming",
                            info="Agent name follows naming conventions",
                            interactive=False
                        )
                with gr.Row():    
                    with gr.Column():
                        gr.Markdown("#### Conciseness Evaluation")
                        gr.Markdown("*Is the description succinct and focused?*")
                        
                        conciseness_concise = gr.Checkbox(
                            label="Is Concise",
                            info="Description is concise without verbose phrasing",
                            interactive=False
                        )
                        conciseness_relevant = gr.Checkbox(
                            label="Is Relevant",
                            info="Description includes only relevant information",
                            interactive=False
                        )
                        conciseness_no_redundancy = gr.Checkbox(
                            label="Avoids Redundancy",
                            info="Description includes no redundant information",
                            interactive=False
                        )
                        token_count_ok = gr.Checkbox(
                            label="Within Token Limit",
                            info="Description is below the max token count",
                            interactive=False
                        )
                        token_count_display = gr.Number(
                            label="Token Count",
                            interactive=False
                        )
            
                # Event handler for evaluation
                def handle_evaluation(agent_desc, max_tokens, template, llm_config):
                    """Handle evaluation and return results for UI update"""
                    feedback, comp, cons, conc = evaluate_description(
                        agent_desc,
                        llm_config,
                        max_tokens,
                        template
                    )
                    
                    return [
                        feedback,  # feedback_output
                        comp.get("contains_name", False),  # completeness_name
                        comp.get("contains_role", False),  # completeness_role
                        comp.get("contains_capabilities", False),  # completeness_capabilities
                        comp.get("contains_examples", False),  # completeness_examples
                        comp.get("contains_edgecases", False),  # completeness_edgecases
                        cons.get("has_conform_structure", False),  # consistency_structure
                        cons.get("has_conform_name", False),  # consistency_name
                        conc.get("is_concise", False),  # conciseness_concise
                        conc.get("is_relevant", False),  # conciseness_relevant
                        conc.get("avoids_redundancy", False),  # conciseness_no_redundancy
                        conc.get("token_count", 0),  # token_count_display
                        conc.get("allowed_token_count", False)  # token_count_ok
                    ]
                
                evaluate_button.click(
                    fn=handle_evaluation,
                    inputs=[
                        eval_agent_description, max_tokens_input, description_template_input,
                        llm_config_state
                    ],
                    outputs=[
                        feedback_output,
                        completeness_name, completeness_role, completeness_capabilities, 
                        completeness_examples, completeness_edgecases,
                        consistency_structure, consistency_name,
                        conciseness_concise, conciseness_relevant, conciseness_no_redundancy,
                        token_count_display, token_count_ok
                    ]
                )

            with gr.Tab("Overlap Analysis"):
                gr.Markdown("## Agent Overlap Analysis")
                gr.Markdown("Analyze overlap between multiple agent descriptions and get guidance on resolving conflicts.")

                gr.Markdown("### Step 1: Provide the agent descriptions that wou want to analzye.")
                
                gr.Markdown("#### Option 1: Select Directory")
                directory_input = gr.Textbox(
                    label="Directory Path",
                    placeholder="Enter path to directory containing agent files (JSON/TXT/YAML)",
                    info="Directory should contain JSON, YAML, or TXT files with descriptions"
                )
                parse_directory_button = gr.Button("Parse Directory", variant="secondary")
                        
                gr.Markdown("#### Option 2: Manual Entry")
                gr.Markdown("You can manually enter or edit agent descriptions below.")
                
                gr.Markdown("#### Agent Descriptions")
                agent_descriptions_input = gr.TextArea(
                    lines=15,
                    label="Agent Descriptions",
                    placeholder="Enter agent descriptions here, or use the directory parser above.\n\nFormat:\n**Agent Name:**\nAgent description here\n\n---\n\n**Another Agent:**\nAnother description here",
                    info="Enter multiple agent descriptions separated by '---' or use the directory parser"
                )
                
                # Event handler for directory parsing
                parse_directory_button.click(
                    fn=parse_directory_for_agents,
                    inputs=[directory_input],
                    outputs=[agent_descriptions_input]
                )
                
                gr.Markdown("### Step 2: Dynamic Overlap Analysis If you want to use dynamic overlap analysis, enter the orchestrator system prompt and the user query.")
                with gr.Accordion(label="Dynamic Overlap Analysis", open=True):
                    # Dynamic analysis needs additional input fields
                    gr.Markdown("#### Orchestrator System Prompt")
                    planning_prompt_input = gr.TextArea(
                        lines=8,
                        value=load_prompts()["orchestrator_system_prompt"],
                        label="Planning System Prompt",
                        placeholder="Enter the orchestrator's planning system prompt here...\n\nExample:\nYou are an orchestrator that selects and coordinates agents to fulfill user requests. Analyze the user query and select the most appropriate agent(s) based on their descriptions and capabilities.",
                        info="The system prompt that guides how the orchestrator selects and coordinates agents"
                    )
                    
                    gr.Markdown("#### User Query to Analyze")
                    user_query_input = gr.TextArea(
                        lines=3,
                        label="User Query",
                        placeholder="Enter a specific user query to analyze for potential conflicts...\n\nExample:\nGenerate a sales report for Q4 2024 and create a forecast for Q1 2025",
                        info="The specific user request that will be processed by the multi-agent system"
                    )
                
                gr.Markdown("### Step 3: Start the overlap analysis.")

                with gr.Row():
                    analyze_button = gr.Button("Analyze Static Overlap", variant="primary")
                    analyze_dynamic_button = gr.Button("Analyze Dynamic Conflicts", variant="primary")
                    agent_graph_button = gr.Button("Create Agent Responsibility Graph", variant="primary")
                    # test_viz_button = gr.Button("Test Visualization", variant="secondary")
                
                # Results area - placed after render function
                overlap_results = gr.Markdown(
                    label="Overlap Analysis Results",
                    value=""""**Click the analyze button to see results.**
                    
                    
                    
                    """
                )
                
                analyze_button.click(
                    fn=analyze_overlap,
                    inputs=[agent_descriptions_input, llm_config_state],
                    outputs=[overlap_results]
                )
                
                analyze_dynamic_button.click(
                    fn=analyze_dynamic_overlap,
                    inputs=[agent_descriptions_input, planning_prompt_input, user_query_input, llm_config_state],
                    outputs=[overlap_results]
                )

                agent_graph_button.click(
                    fn=create_agent_responsibility_graph,
                    inputs=[agent_descriptions_input, llm_config_state],
                    outputs=[overlap_results]
                )

                # test_viz_button.click(
                #     fn=test_graph_visualization,
                #     outputs=[overlap_results]
                # )

    return application


if __name__ == "__main__":
    app = main()
    app.launch(allowed_paths=["./"])