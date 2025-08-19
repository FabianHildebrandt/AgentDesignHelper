## Motivation: Why Good Agent Descriptions Matter

Well-crafted agent descriptions are the foundation of successful multi-agent systems. They serve multiple critical purposes: 
1. They help the orchestrator and other agents to understand what an agent can do and when to use it.
2. They prevent confusion and overlap between different agents.
3. They set clear expectations about capabilities and limitations. 

A good description acts as both a documentation and a crucial factor for reliable task performance.  

## Writing Compelling Agent Role Descriptions

- Create a specific agent name: "Sales data analyst agent" instead of "Data analyst agent". 
- Avoid abbreviations (e.g. HwAgent/ SwAgent)
- Start with a clear role description answering: "What domain does the agent specialize in?" 
    - For example, instead of writing "This agent helps with data analysis," write "The sales data analyst agent is specialized in creating high-quality sales data reports and forecasts." 
-  Describe the agent's capabilities: Include the agent's core competencies and for each competence superficially break down the use cases of the agent.
-  Provide usage scenarios: Exemplary user queries can help to show the distinction between two similar agents.
-  Specify edge cases (if available):

## Designing Proper Tools and Tool Descriptions

- Use concise function names that convey the purpose (e.g. open_json_file)
- Describe the input parameters and the returned results using docstrings/ attributes or decorators (depending on the agent framework)
- Name all function parameters clearly and avoid abbreviations
- Provide examples for each input parameter and if necessary explain the individual components of a parameter
- If possible, provide default values and determine whether the parameter is optional or required
- Include possible limitations like file size limits or functional constraints (e.g. only supports article numbers from 0234 to 0334).

**Exemplary function signature**
```python
def get_article_revenue(article_number : str, year : int | None = None) -> float:
"""Get the revenue of an article by the article number for a fiscal year. 

    Args:
        article_number (str): Article number consisting of the product category (4 digits) and the variant (2 digits), Example: `0234-34`
        year (int): The fiscal year to return the article revenue, if blank the last year's revenue is returned, Example: `2025` 
    
    Returns:
        float : Revenue in USD

"""
```


## Identifying Distinctive Usage Scenarios

Aim for 3-5 diverse user queries that showcase the agent's primary use cases. The goal of the usage scenarios is to show common requests routed to this agent and reduce potential ambiguity compared to other agents.

**Usage scenarios of the sales data analyst agent:**
- "Please help me to visualize the product revenue of the article 0342-24 over the last 5 years."
- "Which increase of the sales volume can I expect for the product 3423-12 in the next three years?."

**Usage scenarios of the product data analyst agent:**
- "What are the compatibility constraints for the product series 0342-23 and 0342-24?"
- "When was the product 0342-12 first introduced into the market?"

## Recognizing and Documenting Edge Cases

Edge cases define the boundaries of your agent's capabilities and help prevent error scenarios and resulting user frustration by setting realistic expectations.

**How to identify edge cases:**
1. **By design**: Review your agent's functions and tools to understand their inherent limitations and scope constraints.
2. **Through testing**: Conduct user-centric testing with challenging scenarios. Put yourself in the user's shoes and try to "break" the system with edge requests.

**Exemplary edges cases of the sales data analyst agent:**
- "Sales database only contains the data of the last 10 years."
- "The sales data analyst can not access sales data of the product series 0123 and 3045".

Well-documented edge cases build trust and help users make informed decisions about when and how to use your agent.
