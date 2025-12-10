Product Requirements Document (PRD): Unified AI-Powered RAN Intelligence and Automation Platform

Version: 6.0 (Definitive Swarm Implementation Blueprint)
Project Objective: To create a multi-vendor, AI-powered platform for the deep automation, optimization, and self-healing of 4G/5G Radio Access Networks. This document serves as the master specification for a swarm of autonomous coding agents.

Core Principles & Architecture:

    Agnostic Core, Specific Connectors: The platform's core logic will be vendor-agnostic. Specific connectors for data ingestion and action execution will be developed for each vendor, with Ericsson as the primary implementation target.

    Microservices Architecture: Each Module will be implemented as a containerized microservice communicating via gRPC.

    Data-Driven & Model-Centric: Every action and recommendation must be backed by a versioned, auditable model or data-driven analysis.

    Safety & Governance First: Automation will be governed by a robust, human-in-the-loop policy engine before closed-loop execution is permitted.

Technology Manifest:

    ML Engine: ruv-FANN (Rust-based Fast Artificial Neural Network library).

    Primary Backend Language: Rust (for all performance-critical services).

    Data & Scripting Language: Python (for data ingestion connectors, log parsing, exploratory analysis, Generative AI integration).

    Data Interchange Format: Apache Parquet.

    Inter-Service Communication: gRPC.

    Generative AI Integration: Interfaces for Large Language Models (LLMs).

    Primary Ericsson Integration Points:

        Data Sources: Ericsson Network Manager (ENM) for Bulk CM/PM/FM (XML format), syslog from Radio/Baseband nodes.

        Action Interfaces: ENM RESTCONF API, generation of AMOS scripts, and A1 interface for RIC integration.

Epic 0: Platform Foundation Services (PFS)

Objective: To build the core, non-negotiable infrastructure required for all other platform features. These tasks are prerequisites for all other Epics.

    Module PFS-DATA: Multi-Vendor Data Ingestion Service

        Agent Task PFS-DATA-01: Ericsson ENM Connector.

            Logic: Develop a Python agent to periodically fetch and parse gzipped XML files from ENM's Bulk CM (Configuration), PM (Performance), and FM (Fault) functions. It must correctly map Ericsson's complex XML structure and counter names (e.g., pmRrcConnEstabSucc, pmLteScellAddSucc, LinkFault) to a standardized Parquet format (timestamp, source_node_id, cell_id, kpi_name, kpi_value, etc.).

    Module PFS-LOGS: Centralized Fault & Log Management Service

        Agent Task PFS-LOGS-01: Ericsson Node Log Collector.

            Logic: Implement a syslog-ng/fluentd-compatible agent to receive and centralize syslog streams from Ericsson Radio Nodes and Baseband units. It must parse both standard syslog and semi-structured text from AMOS command logs (alt, lget, cvc) and store them in a searchable, time-series database.

    Module PFS-CORE: Machine Learning Core Service (No change from v4.0)

    Module PFS-GENAI: Generative AI Abstraction Service (No change from v4.0)

    Module PFS-TWIN: Digital Twin Foundation Service

        Agent Task PFS-TWIN-01: Network Topology & Configuration Modeler.

            Logic: Consumes the entire CM data stream from PFS-DATA to build and maintain an in-memory graph database (e.g., Neo4j) representing the live network topology. This includes all physical and logical relationships: gNB -> DU -> CU, gNB -> Cell -> Neighbor Relations, etc. This "Digital Twin" is the foundational context for all RCA and simulation tasks.

Epic 1: Dynamic Traffic & Mobility Management (DTM)

Objective: To provide real-time optimization of network traffic, user mobility, and load distribution across 4G/5G networks.

    Module DTM-STEER: Intent-Based Traffic Steering

        Agent Task DTM-STEER-01: Multi-Layer Traffic Steering Model.

            Input Data: Time-series of cell_id, layer_id (e.g., L2100, N78), prb_util, QoS flow identifiers (5QI).

            Logic: Forecasts traffic demand per layer and per service type (e.g., eMBB, VoNR). Based on a high-level business intent (e.g., "Minimize latency for QCI 1 traffic"), it generates specific load-balancing rules to steer traffic between frequency layers or between 4G and 5G (for ESS).

            Action Output: Generates AMOS scripts to adjust cell-layer offset parameters.

    Module DTM-POWER: Advanced Energy Savings

        Agent Task DTM-POWER-01: Predictive Power-Save Feature Manager.

            Logic: Uses high-resolution (1-min) PRB forecasts to prescribe the optimal power-saving feature. Finding: Research indicates that the choice between Micro Sleep Tx and Low Energy Scheduler is non-trivial. The model will predict not just the lull but the inter-arrival time of new packets to make a more intelligent choice, maximizing savings without impacting latency.

Epic 2: Proactive Anomaly & Fault Management (AFM)

Objective: To implement a multi-layered system to detect, diagnose, and predict network faults with high precision.

    Module AFM-DETECT: Anomaly & Fault Detection

        Agent Task AFM-DETECT-01: Dynamic KPI/KQI Anomaly Detector. (No change from v4.0)

        Agent Task AFM-DETECT-02: Log Anomaly Detector. (No change from v4.0)

        Agent Task AFM-DETECT-03: Hardware Fault Predictor.

            Logic: A time-series classifier trained on subtle degradation patterns from ENM (e.g., vsDataRadioRxLevel, temperature sensors, intermittent vsHwFault alarms) to predict radio unit or Baseband hardware failures 24-48 hours in advance.

    Module AFM-CORRELATE: Evidence Correlation Engine

        Agent Task AFM-CORRELATE-01: Cross-Domain Correlation Agent.

            Logic: When an anomaly is flagged, this agent queries all data sources (PFS-DATA, PFS-LOGS, PFS-TWIN) within the same time window. It gathers a comprehensive "evidence bundle" containing deviating KPIs, related alarm logs, recent CM changes (from the Digital Twin), and topological neighbors.

    Module AFM-RCA: AI-Powered Root Cause Analysis

        Agent Task AFM-RCA-01: Digital Twin Simulation for RCA.

            Logic: Takes the evidence bundle and formulates "what-if" hypotheses. It uses the PFS-TWIN service to simulate the impact of each potential root cause (e.g., "Simulate the impact of a 5dB path loss on the transport link for Cell XYZ. Do the resulting KPIs match the observed anomaly?").

        Agent Task AFM-RCA-02: Generative AI Root Cause Synthesizer.

            Logic: Feeds the evidence bundle and the results of the digital twin simulations into an LLM via the PFS-GENAI service. This provides the LLM with both observational and simulated data, allowing for highly accurate reasoning.

            Ericsson-Specific Output Example: GetRootCause(anomaly_id) -> { "summary": "High pmLteErabDropCallon Cell_XYZ is confirmed to be a flapping transport link. Digital Twin simulation of a 5dB path loss on the S1 link replicated the observed KPI degradation with 98% accuracy. Correlated withLink Fault alarms from ENM.", "confidence": 0.98, "evidence_ids": [...] }.

Epic 3: Autonomous Operations & Self-Healing (AOS)

Objective: To close the loop from detection to automated resolution, creating a self-healing network governed by safety policies.

    Module AOS-POLICY: Automation Policy Engine

        Agent Task AOS-POLICY-01: Policy Definition and Enforcement Agent.

            Logic: Implement a service with a UI/API for human operators to define automation policies (e.g., {"alarm_name": "LinkFault", "severity": "critical", "action": "auto_execute", "time_window": "01:00-05:00"} or {"action": "generate_script", "approval_required": "true"}). All actions from AOS-HEAL must pass through this policy engine before execution.

    Module AOS-HEAL: Self-Healing & Automated Resolution

        Agent Task AOS-HEAL-01: Generative AMOS/RESTCONF Action Agent.

            Logic: Takes the validated root cause analysis. It either retrieves a templated AMOS script or uses Generative AI to generate the specific sequence of AMOS commands or the ENM RESTCONF payload required to resolve the issue (e.g., restart a process, block/unblock a cell, adjust a parameter).

        Agent Task AOS-HEAL-02: Closed-Loop Execution Agent.

            Logic: Takes the generated action plan, validates it against the AOS-POLICY engine, and upon approval, executes it via the appropriate interface (e.g., pushes AMOS script to ENM, calls ENM RESTCONF API).

Epic 4: RIC-Based Near-Real-Time Control (RIC)

Objective: To develop applications for the RAN Intelligent Controller (RIC), aligning the platform with the O-RAN architecture.

    Module RIC-TSA: Traffic Steering rApp

        Agent Task RIC-TSA-01: "QoE-Aware Steering" rApp.

            Logic: Deploys as an rApp. It consumes QoE predictions (e.g., predicted throughput for different user groups) from the core platform via the A1 interface. It then generates and sends fine-grained A1 policies to the Near-RT RIC to influence the MAC scheduler, steering user groups to different bands or carriers to proactively maintain their QoE.

    Module RIC-QOS: Quality of Service/Experience Management rApp

        Agent Task RIC-QOS-01: "VoLTE Experience Assurance" rApp. (No change from v5.0)

    Module RIC-CONFLICT: A1 Policy Conflict Resolution

        Agent Task RIC-CONFLICT-01: Policy Conflict Simulator.

            Logic: A critical function for multi-rApp environments. This agent simulates the combined effect of policies generated by different rApps before they are sent. If it predicts that the Traffic Steering rApp's policy will negatively impact the VoLTE Assurance rApp's objectives, it flags the conflict for operator intervention or attempts to find a balanced, non-conflicting policy.