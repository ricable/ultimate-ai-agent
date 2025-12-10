---
name: 🤖 Agent Progress Report
about: Daily/weekly progress report from swarm agents
title: 'Agent Report: [Agent Type] - [Date]'
labels: ['agent-report', 'progress']
assignees: ''
---

## 🤖 Agent Information

**Agent Type:** [coordinator/researcher/coder/analyst/tester/architect/etc.]
**Agent ID:** [Unique identifier]
**Epic/Task Reference:** #[EPIC_NUMBER]
**Report Date:** [YYYY-MM-DD]
**Report Type:** [Daily/Weekly/Milestone]

## 📋 Current Assignment

### Primary Task
**Task Description:** [Brief description of current main task]
**Status:** [Not Started/In Progress/Completed/Blocked]
**Priority:** [High/Medium/Low]
**Due Date:** [YYYY-MM-DD]

### Secondary Tasks
- [ ] [Secondary task 1] - [Status]
- [ ] [Secondary task 2] - [Status]
- [ ] [Secondary task 3] - [Status]

## ✅ Accomplishments

### Completed Today/This Week
- [Accomplishment 1 with specific details]
- [Accomplishment 2 with specific details]
- [Accomplishment 3 with specific details]

### Files Created/Modified
- `[filepath]` - [Description of changes]
- `[filepath]` - [Description of changes]
- `[filepath]` - [Description of changes]

### Decisions Made
- **Decision:** [Decision description]
  - **Rationale:** [Why this decision was made]
  - **Impact:** [Effects on project/other agents]
  - **Memory Key:** `decisions/[topic]/[date]`

## 🔄 Current Work

### In Progress
- **Task:** [Current task description]
  - **Progress:** [X% complete or specific progress]
  - **Next Steps:** [What needs to be done next]
  - **Expected Completion:** [Date/timeframe]

### Coordination Activities
- **Agent Interactions:** [Which agents coordinated with]
- **Shared Memory Updates:** [Memory keys updated]
- **Cross-Agent Dependencies:** [Dependencies resolved/created]

## 🚧 Blockers & Challenges

### Current Blockers
- **Blocker 1:** [Description]
  - **Impact:** [How it affects progress]
  - **Potential Solutions:** [Proposed solutions]
  - **Help Needed:** [What assistance is required]

### Technical Challenges
- **Challenge:** [Description]
  - **Status:** [Investigation/Resolution progress]
  - **Solution Approach:** [How being addressed]

## 📊 Performance Metrics

### Task Completion Rate
- **Planned:** [Number of planned tasks]
- **Completed:** [Number completed]
- **Success Rate:** [Percentage]

### Quality Metrics
- **Tests Passed:** [Number/percentage]
- **Code Review Score:** [If applicable]
- **Documentation Coverage:** [Percentage]

### Efficiency Metrics
- **Average Task Duration:** [Time]
- **Rework Rate:** [Percentage]
- **Resource Utilization:** [CPU/Memory usage]

## 🎯 Upcoming Work

### Next 24 Hours
- [ ] [Planned task 1]
- [ ] [Planned task 2]
- [ ] [Planned task 3]

### This Week
- [ ] [Weekly goal 1]
- [ ] [Weekly goal 2]
- [ ] [Weekly goal 3]

### Dependencies
- **Waiting For:** [What agent is waiting for]
- **Blocking Others:** [What others are waiting for from this agent]

## 🧠 Learning & Insights

### New Knowledge Gained
- [Learning 1]
- [Learning 2]
- [Learning 3]

### Process Improvements
- **Improvement:** [Description]
  - **Implementation:** [How to implement]
  - **Expected Benefit:** [What improvement will achieve]

### Pattern Recognition
- **Pattern Observed:** [Description]
- **Optimization Opportunity:** [How to leverage]

## 🔗 Coordination Points

### Memory Updates
- **Key:** `[memory-key]` - **Value:** [Description]
- **Key:** `[memory-key]` - **Value:** [Description]

### Agent Communications
- **Sent Messages:** [Summary of outbound communications]
- **Received Feedback:** [Summary of feedback received]

### Synchronization Points
- **Next Sync:** [When/with whom]
- **Agenda Items:** [What to discuss]

## 📈 Recommendations

### For Project
- [Recommendation 1 for overall project]
- [Recommendation 2 for overall project]

### For Other Agents
- [Recommendation for specific agent type]
- [Recommendation for coordination improvement]

### For Tools/Process
- [Tool improvement suggestion]
- [Process optimization suggestion]

## 🔄 Status Summary

**Overall Status:** [On Track/At Risk/Blocked/Ahead of Schedule]
**Confidence Level:** [High/Medium/Low]
**Next Milestone:** [Description and date]
**Risk Assessment:** [Low/Medium/High risk with details]

---

## 🤖 Agent Coordination Checklist

- [ ] Updated shared memory with latest progress
- [ ] Reviewed dependencies with other agents
- [ ] Documented decisions and rationale
- [ ] Identified any coordination needs
- [ ] Planned next synchronization points

**Memory Storage Commands Used:**
```bash
npx claude-flow@alpha hooks notification --message "[progress summary]"
npx claude-flow@alpha hooks post-edit --file "[modified-file]" --memory-key "agent/[type]/[date]"
```