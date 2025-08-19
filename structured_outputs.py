from pydantic import BaseModel, Field
from typing import Literal, Optional, List

# -- Node Model --
class NodeBase(BaseModel):
    canonical_id: str = Field(..., description="Unique hierarchical canonical ID using namespace path notation (e.g. Agent/Capability/Task/Subtask)")
    label: str = Field(..., description="Local node label without path information")
    type: Literal["agent", "task", "capability"] = Field(..., description="Type of the node")
    description : str = Field("", description="Brief description of the agent role / capability / task") 

# -- Edge Model --
class Edge(BaseModel):
    description : str = Field(..., description="Explanation of the edge/ relation and its type.")
    source: str = Field(..., description="Canonical string ID of the source node")
    target: str = Field(..., description="Canonical string ID of the target node")
    relation: Literal[
        "has_capability",        # Agent → Capability
        "is_redundant", # Similar Task ↔ Similar Task / Similar Agent ↔ Similar Agent
        "has_task",      # Task → Subtask
        "communicates_with", # Agent ↔ Agent
        "manages" # Manager Agent ↔ Managed Agent 
    ] = Field(..., description="Each edges describes a single relation between the source and target node.")

# -- Graph Container --
class AgentGraph(BaseModel):
    name: Optional[str] = Field(None, description="Name or title of the agent graph")
    description: Optional[str] = Field(None, description="High-level overview of the graph purpose")
    nodes: List[NodeBase] = Field(..., description="All nodes in the graph")
    edges: List[Edge] = Field(..., description="Directed edges describing relationships between nodes")

class CompletenessEval(BaseModel):
    """Completeness evaluation: Does the description cover all key content types?"""
    contains_name : bool = Field(description="Description text contains the agent's name.")
    contains_role : bool = Field(description="Describption contains a role description that defines what the agent does and its area of expertise.")
    contains_capabilities : bool = Field(description="Descprition contains a paragraph explaining the agent's capabilities (what it can do and how it can do it).")
    contains_examples: bool = Field(description="Description contains specific usage scenarios of the agent (exemplary user prompts)")
    contains_edgecases: bool = Field(description="Description contains possible limitations/ edges cases to define the scope and constraints of the agent.")

class ConsistencyEval(BaseModel):
    """Consistency evaluation: Does the description follow expected structure and naming conventions?"""
    has_conform_structure : bool = Field(description="The description follows the structure of the provided template: role > capabilities > examples > edge cases and has the same formatting.")
    has_conform_name : bool = Field(description = "The agent name is unique, distinguishable from others, and follows naming conventions of the provided template. If the name contains abbreviations, they should be explained.")

class ConcisenessEval(BaseModel):
    """Conciseness evaluation: Is the description succinct and focused?"""
    is_concise : bool = Field(description="The description is concise and does not contain verbose or redundant phrasing.")
    is_relevant: bool = Field(description="Description includes ONLY information relevant to this agent's role, capabilities, usage examples and edge cases")
    avoids_redundancy : bool = Field(description="Description includes NO redundant information.")
    token_count: Optional[int] = Field(None, description="Token count of the agent description", extra={"exclude": True})
    allowed_token_count: Optional[bool] = Field(None, description="Description is below the max token count")

class AgentDescriptionEval(BaseModel):
    """Agent description evaluation object of a single agent description."""
    feedback: Optional[str] = Field(None, description="Structured feedback regarding the agent description based on the evaluation criteria")
    completeness: CompletenessEval = Field(description="Completeness evaluation of the agent description")
    consistency: ConsistencyEval = Field(description="Consistency evaluation of the agent description")
    conciseness: ConcisenessEval = Field(description="Conciseness evaluation of the agent description")