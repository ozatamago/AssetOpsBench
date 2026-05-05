assetopsbench-1  | You are an AI assistant who makes step-by-step plan to solve a complicated problem under the help of external agents. 
assetopsbench-1  | For each step, make one task followed by one agent-call.
assetopsbench-1  | Each step denoted by #S1, #S2, #S3 ... can be referred to in later steps as a dependency.
assetopsbench-1  | 
assetopsbench-1  | Each step must contain Task, Agent, Dependency and ExpectedOutput. 
assetopsbench-1  | 1. **Task**: A detailed description of what needs to be done in this step. It should include all necessary details and requirements.
assetopsbench-1  | 2. **Agent**: The external agent to be used for solving this task. Agent needs to be selected from the available agents.
assetopsbench-1  | 3. **Dependency**: A list of previous steps (denoted as `#S1`, `#S2`, etc.) that this step depends on. If no previous steps are required, use `None`.
assetopsbench-1  | 4. **ExpectedOutput**: The anticipated result from the agent's execution.
assetopsbench-1  | 
assetopsbench-1  | ## Output Format (Replace '<...>') ##
assetopsbench-1  | 
assetopsbench-1  | ## Step 1
assetopsbench-1  | #Task1: <describe your task here>
assetopsbench-1  | #Agent1: <agent_name>
assetopsbench-1  | #Dependency1: None
assetopsbench-1  | #ExpectedOutput1: <describe the expected output of the call>
assetopsbench-1  | 
assetopsbench-1  | ## Step 2
assetopsbench-1  | #Task2: <describe next task>
assetopsbench-1  | #Agent2: <agent_name>
assetopsbench-1  | #Dependency2: [<you can use #S1 and more to represent previous outputs as a dependency>]
assetopsbench-1  | #ExpectedOutput2: <describe the expected output of the call>
assetopsbench-1  | 
assetopsbench-1  | And so on...
assetopsbench-1  | 
assetopsbench-1  | ## Here are the available agents: ##
assetopsbench-1  | 
assetopsbench-1  | (1) Agent name: IoT Data Download
assetopsbench-1  | Agent description: Can provide information about IoT sites, asset details, sensor data, and retrieve historical data and metadata for various assets and equipment
assetopsbench-1  | Tasks that agent can solve:
assetopsbench-1  | 1. what sites are there
assetopsbench-1  | 2. what assets are at site MAIN
assetopsbench-1  | 3. download sensor data for Chiller 4 at MAIN site
assetopsbench-1  | 4. download asset history for Chiller 4 at MAIN site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00
assetopsbench-1  | 5. merge these JSON files file1.json and file2.json into a single JSON file
assetopsbench-1  | 6. How do I get a list of properties from a JSON file
assetopsbench-1  | 7. I need to read the JSON file 0001.json.
assetopsbench-1  | 8. how do I calculate the start date for last week or past week?
assetopsbench-1  | 
assetopsbench-1  | (2) Agent name: Failure Mode and Sensor Relevancy Expert for Industrial Asset
assetopsbench-1  | Agent description: Can provide information about failure modes, mapping between failure modes and sensors, and can generate machine learning recipes for specific failures
assetopsbench-1  | Tasks that agent can solve:
assetopsbench-1  | 1. List all failure modes of asset Chiller 6.
assetopsbench-1  | 2. List all failure modes of Chiller 6 that can be detected by Chiller 6 Chiller Efficiency.
assetopsbench-1  | 3. What are the relevant sensors that can be used to monitor the loose wiring failure of Chiller 6?If compressor overheating occurs for Chiller 6, which sensor should be prioritized for monitoring this specific failure?
assetopsbench-1  | 4. When compressor motor of a chiller fails, what is the temporal behavior of the power input.
assetopsbench-1  | 5. When power input of Chiller 6 drops, what is the potential failure that causes it?
assetopsbench-1  | 
assetopsbench-1  | (3) Agent name: Time Series Analytics and Forecasting
assetopsbench-1  | Agent description: Can assist with time series analysis, forecasting, anomaly detection, and model selection, and supports pretrained models, context length specifications, and regression tasks for various time series data
assetopsbench-1  | Tasks that agent can solve:
assetopsbench-1  | 1. What types of analysis are supported?
assetopsbench-1  | 2. What pretrained models are available?
assetopsbench-1  | 3. Provide a model checkpoint for the forecasting task.
assetopsbench-1  | 4. Forecast 'Air Handler AIX765 Condenser Water Flow' using data in 'data/tsfm_test_data/chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp.
assetopsbench-1  | 5. Compute anomaly detection for 'Chiller 6 Return Temperature' using data in '/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json'. Use parameter 'Timestamp' as a timestamp.
assetopsbench-1  | 
assetopsbench-1  | (4) Agent name: WorkOrder Agent
assetopsbench-1  | Agent description: The Work Order (WO) agent can retrieve, analyze, and generate work orders for equipment based on historical data, anomalies, alerts, and performance metrics, offering recommendations for preventive and corrective actions, including bundling, prioritization, and predictive maintenance.
assetopsbench-1  | Tasks that agent can solve:
assetopsbench-1  | 1. Can you check if there are any anomalies, and do you think I should create a work order after reviewing the anomalies?"
assetopsbench-1  | 2. I would like to retrieve the preventive work order details for the equipment labeled as CU02013 for the year 2017. Store the result in a file and provide a pointer to the file.
assetopsbench-1  | 3. Get the daily count of the events of alert, anomaly for 2021 for equipment CU02009. Store the result in a file and provide a pointer to the file.
assetopsbench-1  | 
assetopsbench-1  | 
assetopsbench-1  | ## You are going to solve the following complicated problem: ##
assetopsbench-1  | When an anomaly happens for equipment CWC04009, can you recommend top three work orders to address this problem?
assetopsbench-1  | 
assetopsbench-1  | ## Guidelines: ##
assetopsbench-1  | - Task should be something that can be solved by the agent. Task needs to be clear and unambiguous and contain all the information needed to solve it.
assetopsbench-1  | - A plan usually contains less than 5 steps.
assetopsbench-1  | - Only output the generated plan, do not output any other text.
assetopsbench-1  | 
assetopsbench-1  | 
assetopsbench-1  | 
assetopsbench-1  | Output (your generated plan):