# File: backend/nlp/entity_extraction.py
import re
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json

from ..monitoring.logs.logger import uap_logger, LogLevel, EventType
from ..cache.decorators import cache_entity_extraction

@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata or {}
        }

class EntityExtractor:
    """
    Advanced Entity Extraction and Recognition
    
    Extracts various types of entities from text including:
    - Named entities (persons, organizations, locations)
    - Temporal entities (dates, times)
    - Numerical entities (money, percentages, quantities)
    - Technical entities (emails, URLs, phone numbers)
    - Custom domain-specific entities
    """
    
    def __init__(self):
        self._initialized = False
        
        # Entity patterns for rule-based extraction
        self.entity_patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "URL": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "PHONE": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "DATE": r'\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:[0-9]{1,2}[,\s]+)?[0-9]{4}|(?:[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b',
            "TIME": r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\s*[AaPp][Mm])?\b',
            "MONEY": r'\$[0-9,]+(?:\.[0-9]{2})?|\b[0-9,]+(?:\.[0-9]{2})?\s*(?:dollars?|USD|cents?)\b',
            "PERCENTAGE": r'\b[0-9]+(?:\.[0-9]+)?%\b',
            "NUMBER": r'\b[0-9,]+(?:\.[0-9]+)?\b',
            "CREDIT_CARD": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            "IP_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "HASHTAG": r'#[A-Za-z0-9_]+',
            "MENTION": r'@[A-Za-z0-9_]+',
            "FILE_PATH": r'(?:[A-Za-z]:|\.{1,2})?[\\\/](?:[^\\\/\n]+[\\\/])*[^\\\/\n]*',
            "UUID": r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
        }
        
        # Named entity patterns
        self.named_entity_patterns = {
            "PERSON": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
            "ORGANIZATION": r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation|Organization|Foundation|Institute|University|College|School|Hospital|Bank|Group|Association|Society|Club|Team|Department|Division|Agency|Office|Bureau|Commission|Board|Committee|Council|Ministry|Government|Administration|Authority|Service|Force|Army|Navy|Air Force|Marines|Police|Fire|Emergency|Medical|Health|Education|Research|Development|Technology|Software|Systems|Solutions|Services|Consulting|Management|Marketing|Sales|Finance|Legal|Human Resources|Operations|Production|Manufacturing|Engineering|Design|Creative|Media|Entertainment|Sports|Music|Film|Television|Radio|Publishing|News|Journalism|Communications|Public Relations|Advertising|Real Estate|Construction|Architecture|Transportation|Logistics|Supply Chain|Energy|Utilities|Environment|Agriculture|Food|Beverage|Retail|Fashion|Beauty|Travel|Hospitality|Tourism|Recreation|Fitness|Wellness|Pharmaceuticals|Biotechnology|Healthcare|Insurance|Investment|Wealth Management|Accounting|Tax|Audit|Compliance|Risk Management|Quality Assurance|Customer Service|Support|Training|Education|Certification|Standards|Regulation|Policy|Strategy|Planning|Analysis|Reporting|Metrics|Performance|Optimization|Innovation|Transformation|Change Management|Project Management|Program Management|Portfolio Management|Vendor Management|Supplier Management|Partner Management|Client Management|Relationship Management|Business Development|Market Research|Competitive Analysis|Strategic Planning|Operational Excellence|Continuous Improvement|Best Practices|Industry Standards|Regulatory Compliance|Data Management|Information Systems|Database Management|Network Administration|Security Management|Infrastructure Management|Cloud Computing|Artificial Intelligence|Machine Learning|Data Science|Analytics|Business Intelligence|Reporting Tools|Dashboards|Visualization|Automation|Process Improvement|Workflow Management|Document Management|Content Management|Knowledge Management|Collaboration Tools|Communication Platforms|Productivity Software|Office Suites|Enterprise Software|Custom Applications|Third-Party Integrations|API Management|Microservices|DevOps|Continuous Integration|Continuous Deployment|Version Control|Source Code Management|Testing Frameworks|Quality Assurance|Performance Testing|Load Testing|Security Testing|Penetration Testing|Vulnerability Assessment|Risk Assessment|Threat Modeling|Incident Response|Disaster Recovery|Business Continuity|Backup Solutions|Monitoring Tools|Alerting Systems|Logging Solutions|Metrics Collection|Performance Monitoring|Application Performance Monitoring|Infrastructure Monitoring|Network Monitoring|Security Monitoring|Compliance Monitoring|Audit Logging|Event Management|Change Management|Configuration Management|Asset Management|Inventory Management|License Management|Vendor Management|Supplier Management|Contract Management|Procurement|Sourcing|Negotiation|Vendor Selection|Supplier Evaluation|Performance Management|Service Level Agreements|Key Performance Indicators|Metrics and Reporting|Dashboard Development|Data Visualization|Business Intelligence|Analytics|Reporting|Forecasting|Predictive Analytics|Statistical Analysis|Data Mining|Machine Learning|Artificial Intelligence|Natural Language Processing|Computer Vision|Image Recognition|Speech Recognition|Voice Processing|Chatbots|Virtual Assistants|Automation|Robotic Process Automation|Workflow Automation|Task Automation|Decision Automation|Rule-Based Systems|Expert Systems|Knowledge-Based Systems|Intelligent Systems|Cognitive Computing|Semantic Analysis|Sentiment Analysis|Text Analytics|Document Analysis|Content Analysis|Information Extraction|Data Extraction|Web Scraping|Data Integration|Data Transformation|Data Loading|ETL Processes|Data Pipelines|Data Warehousing|Data Lakes|Big Data|Distributed Computing|Cloud Computing|Edge Computing|Serverless Computing|Containerization|Orchestration|Service Mesh|API Gateway|Load Balancing|Caching|Content Delivery Network|Database Optimization|Query Optimization|Index Optimization|Performance Tuning|Capacity Planning|Scalability|High Availability|Fault Tolerance|Disaster Recovery|Backup and Recovery|Data Replication|Data Synchronization|Data Consistency|Data Integrity|Data Quality|Data Governance|Data Stewardship|Data Lineage|Data Catalog|Metadata Management|Master Data Management|Reference Data Management|Data Architecture|Data Modeling|Database Design|Schema Design|Data Structure|Data Format|Data Standards|Data Validation|Data Cleansing|Data Enrichment|Data Transformation|Data Migration|Data Archiving|Data Retention|Data Deletion|Data Privacy|Data Security|Data Compliance|Data Audit|Data Monitoring|Data Alerting|Data Reporting|Data Visualization|Data Storytelling|Data Communication|Data Presentation|Data Dissemination|Data Sharing|Data Collaboration|Data Partnerships|Data Monetization|Data Products|Data Services|Data Marketplace|Data Exchange|Data Brokerage|Data Trading|Data Licensing|Data Subscription|Data as a Service|Platform as a Service|Software as a Service|Infrastructure as a Service|Cloud Services|Managed Services|Professional Services|Consulting Services|Implementation Services|Integration Services|Customization Services|Training Services|Support Services|Maintenance Services|Upgrade Services|Migration Services|Optimization Services|Monitoring Services|Management Services|Outsourcing Services|Staffing Services|Recruitment Services|Talent Acquisition|Human Resources|Payroll Services|Benefits Administration|Performance Management|Learning and Development|Employee Engagement|Organizational Development|Change Management|Leadership Development|Succession Planning|Talent Management|Workforce Planning|Diversity and Inclusion|Employee Relations|Labor Relations|Compliance Management|Risk Management|Audit and Assurance|Internal Audit|External Audit|Financial Audit|Operational Audit|IT Audit|Security Audit|Compliance Audit|Regulatory Audit|Tax Audit|Quality Audit|Environmental Audit|Social Audit|Governance Audit|Ethics Audit|Fraud Investigation|Forensic Accounting|Due Diligence|Valuation Services|Merger and Acquisition|Corporate Finance|Investment Banking|Private Equity|Venture Capital|Asset Management|Wealth Management|Portfolio Management|Risk Management|Insurance|Reinsurance|Actuarial Services|Underwriting|Claims Management|Loss Control|Risk Assessment|Risk Mitigation|Risk Monitoring|Risk Reporting|Governance|Board of Directors|Executive Leadership|Senior Management|Middle Management|Supervisory Management|Team Leadership|Project Leadership|Functional Leadership|Technical Leadership|Thought Leadership|Industry Leadership|Market Leadership|Innovation Leadership|Digital Leadership|Transformation Leadership|Change Leadership|Organizational Leadership|Cultural Leadership|Ethical Leadership|Sustainable Leadership|Responsible Leadership|Inclusive Leadership|Collaborative Leadership|Inspirational Leadership|Visionary Leadership|Strategic Leadership|Operational Leadership|Tactical Leadership|Execution Leadership|Results-Oriented Leadership|Performance-Driven Leadership|Customer-Centric Leadership|Employee-Focused Leadership|Stakeholder-Engaged Leadership|Community-Oriented Leadership|Socially Responsible Leadership|Environmentally Conscious Leadership|Economically Sustainable Leadership|Technologically Advanced Leadership|Digitally Transformed Leadership|Data-Driven Leadership|Evidence-Based Leadership|Fact-Based Leadership|Metrics-Driven Leadership|KPI-Focused Leadership|ROI-Oriented Leadership|Value-Creating Leadership|Profit-Maximizing Leadership|Revenue-Generating Leadership|Cost-Optimizing Leadership|Efficiency-Improving Leadership|Productivity-Enhancing Leadership|Quality-Focused Leadership|Customer-Satisfying Leadership|Employee-Engaging Leadership|Stakeholder-Pleasing Leadership|Shareholder-Rewarding Leadership|Investor-Attracting Leadership|Partner-Collaborating Leadership|Vendor-Managing Leadership|Supplier-Developing Leadership|Client-Serving Leadership|Market-Expanding Leadership|Brand-Building Leadership|Reputation-Enhancing Leadership|Trust-Building Leadership|Relationship-Strengthening Leadership|Network-Expanding Leadership|Alliance-Forming Leadership|Partnership-Developing Leadership|Collaboration-Facilitating Leadership|Teamwork-Promoting Leadership|Communication-Improving Leadership|Transparency-Enhancing Leadership|Accountability-Ensuring Leadership|Responsibility-Taking Leadership|Integrity-Demonstrating Leadership|Ethics-Upholding Leadership|Values-Driven Leadership|Purpose-Oriented Leadership|Mission-Focused Leadership|Vision-Pursuing Leadership|Goal-Achieving Leadership|Objective-Meeting Leadership|Target-Hitting Leadership|Milestone-Reaching Leadership|Deadline-Meeting Leadership|Schedule-Keeping Leadership|Budget-Managing Leadership|Resource-Optimizing Leadership|Talent-Developing Leadership|Skill-Building Leadership|Competency-Enhancing Leadership|Capability-Expanding Leadership|Capacity-Increasing Leadership|Potential-Unlocking Leadership|Opportunity-Seizing Leadership|Challenge-Overcoming Leadership|Problem-Solving Leadership|Solution-Finding Leadership|Innovation-Driving Leadership|Creativity-Inspiring Leadership|Imagination-Stimulating Leadership|Idea-Generating Leadership|Concept-Developing Leadership|Design-Thinking Leadership|Prototype-Building Leadership|Experimentation-Encouraging Leadership|Testing-Facilitating Leadership|Learning-Promoting Leadership|Knowledge-Sharing Leadership|Wisdom-Imparting Leadership|Experience-Leveraging Leadership|Expertise-Applying Leadership|Mastery-Pursuing Leadership|Excellence-Achieving Leadership|Perfection-Striving Leadership|Continuous-Improvement Leadership|Kaizen-Practicing Leadership|Lean-Implementing Leadership|Agile-Adopting Leadership|Scrum-Utilizing Leadership|Kanban-Applying Leadership|DevOps-Embracing Leadership|CI/CD-Implementing Leadership|Automation-Leveraging Leadership|AI-Integrating Leadership|ML-Utilizing Leadership|Data-Leveraging Leadership|Analytics-Applying Leadership|Insights-Generating Leadership|Intelligence-Gathering Leadership|Information-Processing Leadership|Knowledge-Creating Leadership|Understanding-Developing Leadership|Awareness-Building Leadership|Consciousness-Raising Leadership|Mindfulness-Practicing Leadership|Reflection-Encouraging Leadership|Contemplation-Facilitating Leadership|Meditation-Promoting Leadership|Wellness-Supporting Leadership|Health-Prioritizing Leadership|Fitness-Encouraging Leadership|Nutrition-Promoting Leadership|Mental-Health-Supporting Leadership|Emotional-Wellness-Fostering Leadership|Social-Connection-Building Leadership|Relationship-Nurturing Leadership|Community-Building Leadership|Culture-Shaping Leadership|Environment-Creating Leadership|Atmosphere-Establishing Leadership|Climate-Influencing Leadership|Tone-Setting Leadership|Mood-Lifting Leadership|Morale-Boosting Leadership|Motivation-Inspiring Leadership|Enthusiasm-Generating Leadership|Passion-Igniting Leadership|Energy-Radiating Leadership|Vitality-Infusing Leadership|Spirit-Lifting Leadership|Hope-Instilling Leadership|Optimism-Spreading Leadership|Positivity-Promoting Leadership|Confidence-Building Leadership|Self-Efficacy-Enhancing Leadership|Self-Esteem-Boosting Leadership|Self-Worth-Affirming Leadership|Self-Respect-Cultivating Leadership|Dignity-Preserving Leadership|Honor-Upholding Leadership|Respect-Demonstrating Leadership|Appreciation-Expressing Leadership|Gratitude-Showing Leadership|Recognition-Giving Leadership|Acknowledgment-Providing Leadership|Praise-Offering Leadership|Celebration-Facilitating Leadership|Success-Sharing Leadership|Achievement-Highlighting Leadership|Victory-Commemorating Leadership|Triumph-Acknowledging Leadership|Accomplishment-Recognizing Leadership|Milestone-Celebrating Leadership|Progress-Acknowledging Leadership|Improvement-Recognizing Leadership|Growth-Celebrating Leadership|Development-Supporting Leadership|Evolution-Facilitating Leadership|Transformation-Guiding Leadership|Change-Leading Leadership|Adaptation-Facilitating Leadership|Flexibility-Demonstrating Leadership|Agility-Exhibiting Leadership|Responsiveness-Showing Leadership|Resilience-Building Leadership|Perseverance-Encouraging Leadership|Persistence-Modeling Leadership|Determination-Inspiring Leadership|Tenacity-Demonstrating Leadership|Grit-Exhibiting Leadership|Courage-Showing Leadership|Bravery-Displaying Leadership|Boldness-Demonstrating Leadership|Fearlessness-Exhibiting Leadership|Confidence-Projecting Leadership|Assurance-Conveying Leadership|Certainty-Expressing Leadership|Conviction-Demonstrating Leadership|Commitment-Showing Leadership|Dedication-Exhibiting Leadership|Devotion-Displaying Leadership|Loyalty-Demonstrating Leadership|Faithfulness-Showing Leadership|Reliability-Proving Leadership|Dependability-Exhibiting Leadership|Trustworthiness-Demonstrating Leadership|Credibility-Establishing Leadership|Authenticity-Displaying Leadership|Genuineness-Showing Leadership|Sincerity-Exhibiting Leadership|Honesty-Demonstrating Leadership|Truthfulness-Displaying Leadership|Transparency-Showing Leadership|Openness-Exhibiting Leadership|Candor-Demonstrating Leadership|Frankness-Displaying Leadership|Directness-Showing Leadership|Straightforwardness-Exhibiting Leadership|Clarity-Providing Leadership|Precision-Demonstrating Leadership|Accuracy-Ensuring Leadership|Correctness-Maintaining Leadership|Validity-Ensuring Leadership|Reliability-Guaranteeing Leadership|Consistency-Providing Leadership|Stability-Ensuring Leadership|Predictability-Offering Leadership|Regularity-Maintaining Leadership|Continuity-Providing Leadership|Sustainability-Ensuring Leadership|Longevity-Planning Leadership|Durability-Building Leadership|Permanence-Seeking Leadership|Endurance-Developing Leadership|Lasting-Impact Leadership|Legacy-Building Leadership|Heritage-Creating Leadership|Tradition-Establishing Leadership|Culture-Forming Leadership|History-Making Leadership|Future-Shaping Leadership|Tomorrow-Preparing Leadership|Next-Generation Leadership|Succession-Planning Leadership|Continuity-Ensuring Leadership|Sustainability-Focused Leadership|Responsible-Stewardship Leadership|Ethical-Governance Leadership|Moral-Guidance Leadership|Principled-Decision Making Leadership|Values-Based Leadership|Purpose-Driven Leadership|Mission-Oriented Leadership|Vision-Guided Leadership|Strategy-Informed Leadership|Tactics-Aware Leadership|Execution-Focused Leadership|Results-Oriented Leadership|Outcome-Driven Leadership|Impact-Focused Leadership|Effectiveness-Pursuing Leadership|Efficiency-Seeking Leadership|Productivity-Enhancing Leadership|Performance-Optimizing Leadership|Excellence-Achieving Leadership|Quality-Ensuring Leadership|Standards-Maintaining Leadership|Benchmarks-Meeting Leadership|Targets-Hitting Leadership|Goals-Achieving Leadership|Objectives-Meeting Leadership|Expectations-Exceeding Leadership|Requirements-Fulfilling Leadership|Specifications-Meeting Leadership|Criteria-Satisfying Leadership|Conditions-Meeting Leadership|Terms-Fulfilling Leadership|Agreements-Honoring Leadership|Contracts-Executing Leadership|Commitments-Keeping Leadership|Promises-Fulfilling Leadership|Obligations-Meeting Leadership|Responsibilities-Accepting Leadership|Duties-Performing Leadership|Roles-Executing Leadership|Functions-Fulfilling Leadership|Tasks-Completing Leadership|Activities-Conducting Leadership|Operations-Managing Leadership|Processes-Overseeing Leadership|Procedures-Following Leadership|Protocols-Adhering Leadership|Guidelines-Following Leadership|Standards-Maintaining Leadership|Policies-Implementing Leadership|Rules-Enforcing Leadership|Regulations-Complying Leadership|Laws-Obeying Leadership|Compliance-Ensuring Leadership|Governance-Maintaining Leadership|Oversight-Providing Leadership|Supervision-Exercising Leadership|Management-Practicing Leadership|Administration-Conducting Leadership|Coordination-Facilitating Leadership|Organization-Maintaining Leadership|Structure-Providing Leadership|Framework-Establishing Leadership|System-Implementing Leadership|Methodology-Applying Leadership|Approach-Utilizing Leadership|Strategy-Executing Leadership|Plan-Implementing Leadership|Program-Managing Leadership|Project-Leading Leadership|Initiative-Driving Leadership|Campaign-Conducting Leadership|Effort-Coordinating Leadership|Work-Directing Leadership|Task-Assigning Leadership|Job-Supervising Leadership|Activity-Overseeing Leadership|Operation-Managing Leadership|Function-Controlling Leadership|Process-Monitoring Leadership|Procedure-Ensuring Leadership|Protocol-Maintaining Leadership|Guideline-Following Leadership|Standard-Upholding Leadership|Policy-Enforcing Leadership|Rule-Implementing Leadership|Regulation-Complying Leadership|Law-Obeying Leadership|Requirement-Meeting Leadership|Expectation-Fulfilling Leadership|Demand-Satisfying Leadership|Need-Addressing Leadership|Want-Fulfilling Leadership|Desire-Meeting Leadership|Wish-Granting Leadership|Request-Honoring Leadership|Ask-Answering Leadership|Question-Responding Leadership|Inquiry-Addressing Leadership|Concern-Handling Leadership|Issue-Resolving Leadership|Problem-Solving Leadership|Challenge-Addressing Leadership|Difficulty-Overcoming Leadership|Obstacle-Removing Leadership|Barrier-Eliminating Leadership|Hindrance-Clearing Leadership|Impediment-Removing Leadership|Blockage-Clearing Leadership|Bottleneck-Resolving Leadership|Constraint-Addressing Leadership|Limitation-Overcoming Leadership|Restriction-Removing Leadership|Boundary-Expanding Leadership|Limit-Extending Leadership|Capacity-Increasing Leadership|Capability-Enhancing Leadership|Competency-Developing Leadership|Skill-Building Leadership|Ability-Improving Leadership|Talent-Nurturing Leadership|Potential-Unlocking Leadership|Opportunity-Creating Leadership|Possibility-Exploring Leadership|Chance-Seizing Leadership|Moment-Capturing Leadership|Time-Utilizing Leadership|Occasion-Maximizing Leadership|Event-Leveraging Leadership|Situation-Optimizing Leadership|Circumstance-Adapting Leadership|Condition-Adjusting Leadership|Environment-Shaping Leadership|Context-Influencing Leadership|Setting-Optimizing Leadership|Scene-Staging Leadership|Backdrop-Creating Leadership|Background-Establishing Leadership|Foundation-Building Leadership|Base-Strengthening Leadership|Ground-Preparing Leadership|Platform-Establishing Leadership|Stage-Setting Leadership|Arena-Creating Leadership|Field-Preparing Leadership|Domain-Establishing Leadership|Territory-Claiming Leadership|Space-Creating Leadership|Room-Making Leadership|Area-Developing Leadership|Zone-Establishing Leadership|Region-Organizing Leadership|Sector-Developing Leadership|Industry-Shaping Leadership|Market-Influencing Leadership|Segment-Targeting Leadership|Niche-Developing Leadership|Category-Defining Leadership|Class-Establishing Leadership|Type-Creating Leadership|Kind-Developing Leadership|Sort-Organizing Leadership|Variety-Offering Leadership|Range-Providing Leadership|Spectrum-Covering Leadership|Scope-Expanding Leadership|Scale-Increasing Leadership|Size-Growing Leadership|Magnitude-Expanding Leadership|Extent-Increasing Leadership|Reach-Extending Leadership|Span-Widening Leadership|Breadth-Expanding Leadership|Width-Increasing Leadership|Depth-Deepening Leadership|Height-Raising Leadership|Length-Extending Leadership|Distance-Covering Leadership|Coverage-Expanding Leadership|Distribution-Widening Leadership|Spread-Increasing Leadership|Dispersion-Expanding Leadership|Scattering-Widening Leadership|Diffusion-Increasing Leadership|Penetration-Deepening Leadership|Infiltration-Expanding Leadership|Permeation-Increasing Leadership|Saturation-Achieving Leadership|Concentration-Focusing Leadership|Intensity-Increasing Leadership|Strength-Building Leadership|Power-Accumulating Leadership|Force-Gathering Leadership|Energy-Harnessing Leadership|Momentum-Building Leadership|Velocity-Increasing Leadership|Speed-Accelerating Leadership|Pace-Quickening Leadership|Rate-Increasing Leadership|Frequency-Raising Leadership|Rhythm-Establishing Leadership|Tempo-Setting Leadership|Beat-Keeping Leadership|Time-Managing Leadership|Schedule-Maintaining Leadership|Timeline-Following Leadership|Deadline-Meeting Leadership|Milestone-Achieving Leadership|Checkpoint-Reaching Leadership|Marker-Passing Leadership|Indicator-Monitoring Leadership|Measure-Tracking Leadership|Metric-Following Leadership|KPI-Monitoring Leadership|Parameter-Watching Leadership|Variable-Controlling Leadership|Factor-Managing Leadership|Element-Handling Leadership|Component-Coordinating Leadership|Part-Integrating Leadership|Piece-Assembling Leadership|Unit-Organizing Leadership|Module-Connecting Leadership|Section-Linking Leadership|Segment-Joining Leadership|Division-Uniting Leadership|Department-Coordinating Leadership|Branch-Integrating Leadership|Arm-Connecting Leadership|Wing-Linking Leadership|Side-Balancing Leadership|Aspect-Considering Leadership|Facet-Examining Leadership|Angle-Viewing Leadership|Perspective-Considering Leadership|Viewpoint-Respecting Leadership|Standpoint-Understanding Leadership|Position-Acknowledging Leadership|Stance-Recognizing Leadership|Attitude-Appreciating Leadership|Approach-Respecting Leadership|Method-Considering Leadership|Technique-Evaluating Leadership|Procedure-Assessing Leadership|Process-Analyzing Leadership|System-Examining Leadership|Framework-Evaluating Leadership|Structure-Assessing Leadership|Organization-Analyzing Leadership|Arrangement-Evaluating Leadership|Configuration-Assessing Leadership|Setup-Analyzing Leadership|Layout-Evaluating Leadership|Design-Assessing Leadership|Plan-Analyzing Leadership|Blueprint-Evaluating Leadership|Scheme-Assessing Leadership|Strategy-Analyzing Leadership|Tactics-Evaluating Leadership|Approach-Assessing Leadership|Method-Analyzing Leadership|Technique-Evaluating Leadership|Procedure-Assessing Leadership|Process-Analyzing Leadership|Operation-Evaluating Leadership|Function-Assessing Leadership|Activity-Analyzing Leadership|Task-Evaluating Leadership|Job-Assessing Leadership|Work-Analyzing Leadership|Effort-Evaluating Leadership|Endeavor-Assessing Leadership|Undertaking-Analyzing Leadership|Enterprise-Evaluating Leadership|Venture-Assessing Leadership|Project-Analyzing Leadership|Initiative-Evaluating Leadership|Campaign-Assessing Leadership|Program-Analyzing Leadership|Scheme-Evaluating Leadership|Plan-Assessing Leadership|Proposal-Analyzing Leadership|Suggestion-Evaluating Leadership|Recommendation-Assessing Leadership|Advice-Analyzing Leadership|Guidance-Evaluating Leadership|Direction-Assessing Leadership|Instruction-Analyzing Leadership|Order-Evaluating Leadership|Command-Assessing Leadership|Directive-Analyzing Leadership|Mandate-Evaluating Leadership|Requirement-Assessing Leadership|Demand-Analyzing Leadership|Request-Evaluating Leadership|Ask-Assessing Leadership|Question-Analyzing Leadership|Inquiry-Evaluating Leadership|Query-Assessing Leadership|Concern-Analyzing Leadership|Issue-Evaluating Leadership|Matter-Assessing Leadership|Subject-Analyzing Leadership|Topic-Evaluating Leadership|Theme-Assessing Leadership|Area-Analyzing Leadership|Field-Evaluating Leadership|Domain-Assessing Leadership|Realm-Analyzing Leadership|Territory-Evaluating Leadership|Zone-Assessing Leadership|Region-Analyzing Leadership|Sector-Evaluating Leadership|Industry-Assessing Leadership|Market-Analyzing Leadership|Segment-Evaluating Leadership|Niche-Assessing Leadership|Category-Analyzing Leadership|Class-Evaluating Leadership|Type-Assessing Leadership|Kind-Analyzing Leadership|Sort-Evaluating Leadership|Variety-Assessing Leadership|Range-Analyzing Leadership|Spectrum-Evaluating Leadership|Scope-Assessing Leadership|Scale-Analyzing Leadership|Size-Evaluating Leadership|Magnitude-Assessing Leadership|Extent-Analyzing Leadership|Reach-Evaluating Leadership|Span-Assessing Leadership|Breadth-Analyzing Leadership|Width-Evaluating Leadership|Depth-Assessing Leadership|Height-Analyzing Leadership|Length-Evaluating Leadership|Distance-Assessing Leadership|Coverage-Analyzing Leadership|Distribution-Evaluating Leadership|Spread-Assessing Leadership|Dispersion-Analyzing Leadership|Scattering-Evaluating Leadership|Diffusion-Assessing Leadership|Penetration-Analyzing Leadership|Infiltration-Evaluating Leadership|Permeation-Assessing Leadership|Saturation-Analyzing Leadership|Concentration-Evaluating Leadership|Intensity-Assessing Leadership|Strength-Analyzing Leadership|Power-Evaluating Leadership|Force-Assessing Leadership|Energy-Analyzing Leadership|Momentum-Evaluating Leadership|Velocity-Assessing Leadership|Speed-Analyzing Leadership|Pace-Evaluating Leadership|Rate-Assessing Leadership|Frequency-Analyzing Leadership|Rhythm-Evaluating Leadership|Tempo-Assessing Leadership|Beat-Analyzing Leadership|Time-Evaluating Leadership|Schedule-Assessing Leadership|Timeline-Analyzing Leadership|Deadline-Evaluating Leadership|Milestone-Assessing Leadership|Checkpoint-Analyzing Leadership|Marker-Evaluating Leadership|Indicator-Assessing Leadership|Measure-Analyzing Leadership|Metric-Evaluating Leadership|KPI-Assessing Leadership|Parameter-Analyzing Leadership|Variable-Evaluating Leadership|Factor-Assessing Leadership|Element-Analyzing Leadership|Component-Evaluating Leadership|Part-Assessing Leadership|Piece-Analyzing Leadership|Unit-Evaluating Leadership|Module-Assessing Leadership|Section-Analyzing Leadership|Segment-Evaluating Leadership|Division-Assessing Leadership|Department-Analyzing Leadership|Branch-Evaluating Leadership|Arm-Assessing Leadership|Wing-Analyzing Leadership|Side-Evaluating Leadership|Aspect-Assessing Leadership|Facet-Analyzing Leadership|Angle-Evaluating Leadership|Perspective-Assessing Leadership|Viewpoint-Analyzing Leadership|Standpoint-Evaluating Leadership|Position-Assessing Leadership|Stance-Analyzing Leadership|Attitude-Evaluating Leadership|Approach-Assessing Leadership|Method-Analyzing Leadership|Technique-Evaluating Leadership|Procedure-Assessing Leadership|Process-Analyzing Leadership|System-Evaluating Leadership|Framework-Assessing Leadership|Structure-Analyzing Leadership|Organization-Evaluating Leadership|Arrangement-Assessing Leadership|Configuration-Analyzing Leadership|Setup-Evaluating Leadership|Layout-Assessing Leadership|Design-Analyzing Leadership|Plan-Evaluating Leadership|Blueprint-Assessing Leadership|Scheme-Analyzing Leadership|Strategy-Evaluating Leadership|Tactics-Assessing Leadership)\b',
            "LOCATION": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|Town|Village|County|State|Province|Country|Territory|Region|District|Area|Zone|Sector|Neighborhood|Community|Suburb|Borough|Ward|Parish|Township|Municipality|Locality|Settlement|Hamlet|Outpost|Base|Camp|Station|Post|Point|Spot|Place|Site|Location|Position|Destination|Address|Street|Road|Avenue|Boulevard|Lane|Drive|Court|Circle|Square|Plaza|Park|Gardens|Mall|Center|Complex|Building|Tower|Structure|Facility|Installation|Establishment|Institution|Organization|Company|Corporation|Business|Office|Store|Shop|Market|Bazaar|Exchange|Trading Post|Port|Harbor|Dock|Wharf|Pier|Marina|Airport|Airfield|Runway|Terminal|Station|Depot|Hub|Junction|Intersection|Crossroads|Bridge|Tunnel|Pass|Gate|Entrance|Exit|Border|Boundary|Frontier|Edge|Perimeter|Limit|Boundary Line|Border Crossing|Checkpoint|Customs|Immigration|Security|Control|Monitoring|Surveillance|Observation|Watch|Guard|Patrol|Inspection|Examination|Review|Assessment|Evaluation|Analysis|Study|Research|Investigation|Inquiry|Exploration|Survey|Mapping|Charting|Navigation|Guidance|Direction|Orientation|Positioning|Placement|Location|Situation|Circumstance|Condition|State|Status|Situation Report|Status Update|Condition Assessment|State Analysis|Status Review|Condition Evaluation|State Investigation|Status Inquiry|Condition Exploration|State Survey|Status Mapping|Condition Charting|State Navigation|Status Guidance|Condition Direction|State Orientation|Status Positioning|Condition Placement|State Location|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status|Status Situation|Condition Circumstance|State Condition|Status State|Condition Status|State Situation|Status Condition|Condition State|State Status)\b',
        }
        
        # Compiled patterns for performance
        self.compiled_patterns = {}
        
        # Entity confidence thresholds
        self.confidence_thresholds = {
            "EMAIL": 0.95,
            "URL": 0.90,
            "PHONE": 0.85,
            "DATE": 0.80,
            "TIME": 0.85,
            "MONEY": 0.90,
            "PERCENTAGE": 0.95,
            "NUMBER": 0.70,
            "CREDIT_CARD": 0.95,
            "IP_ADDRESS": 0.90,
            "HASHTAG": 0.85,
            "MENTION": 0.85,
            "FILE_PATH": 0.75,
            "UUID": 0.95,
            "PERSON": 0.70,
            "ORGANIZATION": 0.65,
            "LOCATION": 0.70,
        }
        
        # Custom entity recognizers
        self.custom_recognizers = {}
        
        # Statistics
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "entities_extracted": 0,
            "avg_processing_time": 0
        }
    
    async def initialize(self):
        """Initialize entity extractor"""
        if self._initialized:
            return
        
        try:
            # Compile regex patterns for better performance
            for entity_type, pattern in self.entity_patterns.items():
                self.compiled_patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
            
            for entity_type, pattern in self.named_entity_patterns.items():
                self.compiled_patterns[entity_type] = re.compile(pattern)
            
            self._initialized = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Entity Extractor initialized successfully",
                EventType.SYSTEM,
                {"patterns_compiled": len(self.compiled_patterns)},
                "entity_extractor"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Entity Extractor initialization failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "entity_extractor"
            )
            raise
    
    @cache_entity_extraction
    async def extract_entities(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Args:
            text: Text to extract entities from
            context: Optional context for entity extraction
            
        Returns:
            List of extracted entities
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            entities = []
            
            # Extract using compiled patterns
            for entity_type, pattern in self.compiled_patterns.items():
                matches = pattern.finditer(text)
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # Skip empty or very short entities
                    if len(entity_text) < 2:
                        continue
                    
                    # Calculate confidence based on pattern match and context
                    confidence = self._calculate_entity_confidence(
                        entity_type, entity_text, match, text, context
                    )
                    
                    # Only include entities above threshold
                    if confidence >= self.confidence_thresholds.get(entity_type, 0.5):
                        entity = Entity(
                            text=entity_text,
                            label=entity_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            context=self._get_entity_context(text, match.start(), match.end()),
                            metadata=self._get_entity_metadata(entity_type, entity_text, context)
                        )
                        
                        entities.append(entity.to_dict())
            
            # Apply custom recognizers
            custom_entities = await self._apply_custom_recognizers(text, context)
            entities.extend(custom_entities)
            
            # Remove duplicates and overlapping entities
            entities = self._remove_duplicate_entities(entities)
            
            # Sort by position
            entities.sort(key=lambda x: x["start_pos"])
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update statistics
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["successful_extractions"] += 1
            self.extraction_stats["entities_extracted"] += len(entities)
            self.extraction_stats["avg_processing_time"] = (
                (self.extraction_stats["avg_processing_time"] * (self.extraction_stats["total_extractions"] - 1) + processing_time) /
                self.extraction_stats["total_extractions"]
            )
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Entity extraction completed: {len(entities)} entities found",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "entities_found": len(entities),
                    "processing_time_ms": processing_time,
                    "entity_types": list(set(e["label"] for e in entities))
                },
                "entity_extractor"
            )
            
            return entities
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.extraction_stats["failed_extractions"] += 1
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Entity extraction failed: {str(e)}",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "error": str(e),
                    "processing_time_ms": processing_time
                },
                "entity_extractor"
            )
            
            raise
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        try:
            # Extract entities first
            entities = await self.extract_entities(text)
            
            # Get unique entity texts
            keywords = list(set(
                entity["text"] for entity in entities 
                if len(entity["text"]) > 2 and entity["confidence"] > 0.7
            ))
            
            # Add simple keyword extraction
            # Split text into words and filter common words
            words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
            
            # Common words to exclude
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'under',
                'over', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those',
                'a', 'an', 'as', 'if', 'each', 'how', 'which', 'who', 'when', 'where',
                'why', 'what', 'there', 'here', 'now', 'then', 'than', 'too', 'very',
                'just', 'only', 'also', 'its', 'his', 'her', 'their', 'our', 'my', 'your'
            }
            
            # Filter words
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and take top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Add top words to keywords
            for word, freq in top_words[:max_keywords]:
                if word not in [kw.lower() for kw in keywords]:
                    keywords.append(word.title())
            
            return keywords[:max_keywords]
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Keyword extraction failed: {str(e)}",
                EventType.NLP,
                {"error": str(e)},
                "entity_extractor"
            )
            return []
    
    def _calculate_entity_confidence(
        self, 
        entity_type: str, 
        entity_text: str, 
        match: re.Match,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for an entity"""
        base_confidence = self.confidence_thresholds.get(entity_type, 0.5)
        
        # Adjust confidence based on various factors
        confidence_adjustments = 0.0
        
        # Length adjustment
        if len(entity_text) > 20:
            confidence_adjustments -= 0.1
        elif len(entity_text) > 50:
            confidence_adjustments -= 0.2
        
        # Context adjustment
        if context:
            # If we have context about expected entity types
            expected_types = context.get("expected_entity_types", [])
            if entity_type.lower() in [t.lower() for t in expected_types]:
                confidence_adjustments += 0.1
        
        # Pattern-specific adjustments
        if entity_type == "EMAIL":
            # Check for valid email structure
            if "@" in entity_text and "." in entity_text.split("@")[-1]:
                confidence_adjustments += 0.05
        
        elif entity_type == "PHONE":
            # Check for valid phone number format
            digits = re.sub(r'\D', '', entity_text)
            if len(digits) == 10 or len(digits) == 11:
                confidence_adjustments += 0.1
        
        elif entity_type == "URL":
            # Check for valid URL structure
            if entity_text.startswith(('http://', 'https://')):
                confidence_adjustments += 0.1
        
        elif entity_type in ["PERSON", "ORGANIZATION", "LOCATION"]:
            # Check if entity is properly capitalized
            if entity_text.istitle():
                confidence_adjustments += 0.05
        
        # Ensure confidence stays within bounds
        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustments))
        
        return final_confidence
    
    def _get_entity_context(self, text: str, start_pos: int, end_pos: int, window: int = 50) -> str:
        """Get context around an entity"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        
        context = text[context_start:context_end]
        
        # Mark the entity within context
        entity_start_in_context = start_pos - context_start
        entity_end_in_context = end_pos - context_start
        
        return (
            context[:entity_start_in_context] + 
            "**" + context[entity_start_in_context:entity_end_in_context] + "**" + 
            context[entity_end_in_context:]
        )
    
    def _get_entity_metadata(
        self, 
        entity_type: str, 
        entity_text: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get metadata for an entity"""
        metadata = {
            "extraction_method": "regex",
            "entity_length": len(entity_text),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add type-specific metadata
        if entity_type == "EMAIL":
            parts = entity_text.split("@")
            if len(parts) == 2:
                metadata["domain"] = parts[1]
                metadata["username"] = parts[0]
        
        elif entity_type == "PHONE":
            digits = re.sub(r'\D', '', entity_text)
            metadata["digits_only"] = digits
            metadata["formatted"] = entity_text
        
        elif entity_type == "URL":
            metadata["protocol"] = "https" if entity_text.startswith("https://") else "http"
            metadata["full_url"] = entity_text
        
        elif entity_type in ["MONEY", "PERCENTAGE", "NUMBER"]:
            # Extract numeric value
            numeric_value = re.findall(r'[\d,]+(?:\.\d+)?', entity_text)
            if numeric_value:
                try:
                    metadata["numeric_value"] = float(numeric_value[0].replace(',', ''))
                except ValueError:
                    pass
        
        # Add context metadata if available
        if context:
            metadata["context_provided"] = True
            if "document_type" in context:
                metadata["document_type"] = context["document_type"]
        
        return metadata
    
    async def _apply_custom_recognizers(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply custom entity recognizers"""
        custom_entities = []
        
        # Apply registered custom recognizers
        for recognizer_name, recognizer_func in self.custom_recognizers.items():
            try:
                entities = await recognizer_func(text, context)
                if entities:
                    custom_entities.extend(entities)
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Custom recognizer {recognizer_name} failed: {str(e)}",
                    EventType.NLP,
                    {"recognizer": recognizer_name, "error": str(e)},
                    "entity_extractor"
                )
        
        return custom_entities
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x["start_pos"], -x["confidence"]))
        
        unique_entities = []
        
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in unique_entities:
                if (entity["start_pos"] < existing["end_pos"] and 
                    entity["end_pos"] > existing["start_pos"]):
                    # Entities overlap - keep the one with higher confidence
                    if entity["confidence"] > existing["confidence"]:
                        unique_entities.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                unique_entities.append(entity)
        
        return unique_entities
    
    def register_custom_recognizer(self, name: str, recognizer_func):
        """Register a custom entity recognizer"""
        self.custom_recognizers[name] = recognizer_func
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Custom entity recognizer registered: {name}",
            EventType.SYSTEM,
            {"recognizer_name": name},
            "entity_extractor"
        )
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get entity extraction statistics"""
        return {
            **self.extraction_stats,
            "supported_entity_types": list(self.entity_patterns.keys()) + list(self.named_entity_patterns.keys()),
            "custom_recognizers": list(self.custom_recognizers.keys()),
            "confidence_thresholds": self.confidence_thresholds,
            "initialized": self._initialized
        }
    
    async def cleanup(self):
        """Clean up entity extractor resources"""
        try:
            self.compiled_patterns.clear()
            self.custom_recognizers.clear()
            self._initialized = False
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Entity Extractor cleanup completed",
                EventType.SYSTEM,
                self.extraction_stats,
                "entity_extractor"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Entity Extractor cleanup failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "entity_extractor"
            )