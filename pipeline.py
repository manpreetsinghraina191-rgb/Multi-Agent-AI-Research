from agents import build_reader_agent,build_search_agent,writer_chain,critic_chain


def run_research_pipeline(topic : str)-> dict:
    state={}
    # search agent working 
    print("\n" + "*"*50)
    print("step 1 - search agent is working ...............")
    search_agent=build_search_agent()
    search_result=search_agent.invoke({
        "messages":  [("user",f"Find recent reliable and detailed information bout :{topic}")]
    })
    state["search_results"]=search_result['messages'][-1].content
    print("\n   search result   ", state['search_results'])

    print("\n" + "*"*50)
    print("step 2 - reader agent is scraping top resources ...............")
    reader_agent= build_reader_agent()
    reader_result=reader_agent.invoke({
         "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
         )]
    })
    state['scraped_content']=reader_result['messages'][-1].content
    print("scraped content\n", state['scraped_content'])

    # step3 - writer chain

    print("\n"+ "*"*50)
    print("step 3 -- Writer is drafting the report ...........")
    research_combined=(
        f" Search results :  \n {state['search_results']}\n\n\n"
        f"Detailed scraped content  : \n { state['scraped_content']}"
    )
    state['report']=writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n\n Final Report \n\n",state['report'])

    #critic report

    print("\n" + "*"*50)
    print("step 3 - Critic is reviewing the final report  ...............")
    state['feedback']=critic_chain.invoke({
        "report" : state['report']
    })

    print("\n critic report \n ",state['feedback'])
    return state

if __name__=="__main__":
    topic=input("\n  Enter research topic ")
    run_research_pipeline(topic)