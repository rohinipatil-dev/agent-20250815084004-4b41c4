import streamlit as st
from datetime import datetime, date
import json
import uuid
import re
from typing import List, Dict, Any, Optional
import os

from openai import OpenAI

# Initialize OpenAI client (requires OPENAI_API_KEY env var)
client = OpenAI()

# ----------------------------
# Utilities
# ----------------------------

def extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON object or array from a text that may contain prose.
    Returns parsed JSON if found, else None.
    """
    if not text:
        return None
    # Try to find the first JSON object or array block
    json_pattern = re.compile(r'(\{.*\}|\[.*\])', re.DOTALL)
    match = json_pattern.search(text)
    if not match:
        return None
    block = match.group(1).strip()
    try:
        return json.loads(block)
    except Exception:
        # Attempt to fix common trailing commas
        try:
            block2 = re.sub(r',(\s*[}\]])', r'\1', block)
            return json.loads(block2)
        except Exception:
            return None

def call_openai(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1200) -> str:
    """
    Calls OpenAI chat completion with gpt-4 and returns the content string.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

def new_task(title: str, details: str = "", priority: str = "Medium", due: Optional[date] = None) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "details": details.strip(),
        "priority": priority,
        "due_date": due.isoformat() if due else "",
        "status": "todo",
        "created_at": datetime.utcnow().isoformat(),
        "subtasks": [],
    }

def ensure_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = "All"
    if "sort_by" not in st.session_state:
        st.session_state.sort_by = "Priority"
    if "search" not in st.session_state:
        st.session_state.search = ""
    if "api_ready" not in st.session_state:
        st.session_state.api_ready = bool(os.getenv("OPENAI_API_KEY"))

def add_task_to_state(task: Dict[str, Any]):
    st.session_state.tasks.append(task)

def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    for t in st.session_state.tasks:
        if t["id"] == task_id:
            return t
    return None

def delete_task(task_id: str):
    st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task_id]

def toggle_task_status(task_id: str):
    task = get_task_by_id(task_id)
    if not task:
        return
    task["status"] = "done" if task["status"] != "done" else "todo"

def update_task(task_id: str, title: str, details: str, priority: str, due_date: Optional[date]):
    task = get_task_by_id(task_id)
    if not task:
        return
    task["title"] = title.strip()
    task["details"] = details.strip()
    task["priority"] = priority
    task["due_date"] = due_date.isoformat() if due_date else ""

def clear_completed():
    st.session_state.tasks = [t for t in st.session_state.tasks if t["status"] != "done"]

def sort_tasks(tasks: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    def priority_rank(p: str) -> int:
        mapping = {"High": 0, "Medium": 1, "Low": 2}
        return mapping.get(p, 1)
    if sort_by == "Priority":
        return sorted(tasks, key=lambda x: (priority_rank(x.get("priority", "Medium")), x.get("title", "")))
    if sort_by == "Due date":
        def due_key(d):
            try:
                return datetime.fromisoformat(d["due_date"]) if d.get("due_date") else datetime.max
            except Exception:
                return datetime.max
        return sorted(tasks, key=lambda x: due_key(x))
    if sort_by == "Status":
        return sorted(tasks, key=lambda x: x.get("status", "todo"))
    return tasks

# ----------------------------
# AI helpers
# ----------------------------

def ai_generate_tasks_from_goal(goal: str) -> List[Dict[str, Any]]:
    prompt = f"""
You will create a concise, actionable to-do list from the user's goal.
Return STRICT JSON only, no extra text.

Schema:
{{
  "tasks": [
    {{
      "title": "short action-oriented task title",
      "details": "1-2 sentence detail or acceptance criteria",
      "priority": "High|Medium|Low",
      "due_in_days": number | null
    }}
  ]
}}

Guidelines:
- Prefer 5-10 tasks maximum.
- Titles must be clear and start with a verb.
- Use priority to reflect impact/urgency.
- Use due_in_days only if obviously implied by the goal; else null.

User goal:
{goal}
"""
    content = call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    data = extract_json(content)
    if not data or "tasks" not in data or not isinstance(data["tasks"], list):
        st.warning("Could not parse AI task suggestions.")
        return []
    tasks_out = []
    for t in data["tasks"]:
        title = str(t.get("title", "")).strip()
        if not title:
            continue
        details = str(t.get("details", "")).strip()
        priority = str(t.get("priority", "Medium")).title()
        if priority not in ["High", "Medium", "Low"]:
            priority = "Medium"
        due_days = t.get("due_in_days", None)
        due_dt = None
        try:
            if isinstance(due_days, (int, float)) and due_days >= 0:
                due_dt = date.today().fromordinal(date.today().toordinal() + int(due_days))
        except Exception:
            due_dt = None
        tasks_out.append(new_task(title, details, priority, due_dt))
    return tasks_out

def ai_prioritize_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Provide minimal info for prioritization
    payload = [{"id": t["id"], "title": t["title"], "details": t["details"], "priority": t["priority"], "status": t["status"], "due_date": t["due_date"]} for t in tasks]
    prompt = f"""
Re-prioritize and order the following tasks. Return STRICT JSON only.

Input tasks:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "ordered": [
    {{
      "id": "existing id",
      "priority": "High|Medium|Low"
    }}
  ]
}}

Rules:
- Preserve ids.
- Set a reasonable priority for each.
- 'ordered' array must include all tasks in desired execution order (top = first).
"""
    content = call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    data = extract_json(content)
    if not data or "ordered" not in data or not isinstance(data["ordered"], list):
        st.warning("Could not parse AI prioritization.")
        return tasks
    # Map updates
    priority_map = {}
    order_list = []
    for item in data["ordered"]:
        tid = item.get("id")
        pr = str(item.get("priority", "Medium")).title()
        if pr not in ["High", "Medium", "Low"]:
            pr = "Medium"
        if tid:
            priority_map[tid] = pr
            order_list.append(tid)
    id_to_task = {t["id"]: t for t in tasks}
    # Apply priorities
    for tid, pr in priority_map.items():
        if tid in id_to_task:
            id_to_task[tid]["priority"] = pr
    # Build ordered list keeping any missing at end
    ordered_tasks = [id_to_task[tid] for tid in order_list if tid in id_to_task]
    missing = [t for t in tasks if t["id"] not in set(order_list)]
    return ordered_tasks + missing

def ai_breakdown_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = f"""
Break down the task into 3-7 sequential, actionable subtasks.
Return STRICT JSON only.

Task:
{json.dumps({"title": task["title"], "details": task.get("details","")}, ensure_ascii=False)}

Output schema:
{{
  "subtasks": [
    {{
      "title": "short subtask title",
      "details": "1 sentence detail",
      "priority": "High|Medium|Low"
    }}
  ]
}}
"""
    content = call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    data = extract_json(content)
    if not data or "subtasks" not in data or not isinstance(data["subtasks"], list):
        st.warning("Could not parse AI breakdown.")
        return []
    subs = []
    for s in data["subtasks"]:
        title = str(s.get("title", "")).strip()
        if not title:
            continue
        details = str(s.get("details", "")).strip()
        pr = str(s.get("priority", "Medium")).title()
        if pr not in ["High", "Medium", "Low"]:
            pr = "Medium"
        subs.append({"title": title, "details": details, "priority": pr, "status": "todo"})
    return subs

def ai_daily_plan(tasks: List[Dict[str, Any]], work_hours: int = 6) -> Dict[str, Any]:
    payload = [{"title": t["title"], "details": t["details"], "priority": t["priority"], "status": t["status"], "due_date": t["due_date"]} for t in tasks]
    prompt = f"""
Create a focused plan for today from the tasks below. Assume {work_hours} hours available.
Return STRICT JSON only.

Tasks:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "today_plan": [
    {{
      "title": "task title",
      "reason": "why it's selected",
      "time_estimate_minutes": number
    }}
  ],
  "notes": "1-3 bullet points summary"
}}

Rules:
- Choose only the most impactful tasks that fit the time.
- Sum of time_estimate_minutes should be <= {work_hours*60}.
- Prefer High priority and urgent tasks.
"""
    content = call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    data = extract_json(content) or {}
    return data

# ----------------------------
# UI Components
# ----------------------------

def sidebar_add_task():
    st.sidebar.subheader("Add Task")
    with st.sidebar.form("add_task_form", clear_on_submit=True):
        title = st.text_input("Title", key="add_title")
        details = st.text_area("Details", key="add_details")
        priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1, key="add_priority")
        due = st.date_input("Due date", value=None, format="YYYY-MM-DD", key="add_due")
        submitted = st.form_submit_button("Add")
        if submitted:
            if title.strip():
                due_date = due if isinstance(due, date) else None
                add_task_to_state(new_task(title, details, priority, due_date))
                st.success("Task added.")
            else:
                st.warning("Title is required to add a task.")

def sidebar_ai_tools():
    st.sidebar.subheader("AI Assistant")
    if not st.session_state.api_ready:
        st.sidebar.info("Set OPENAI_API_KEY in your environment to enable AI features.")
    goal = st.sidebar.text_area("Goal or description")
    col1, col2 = st.sidebar.columns(2)
    gen_clicked = col1.button("Generate tasks", use_container_width=True, disabled=not st.session_state.api_ready)
    pri_clicked = col2.button("Prioritize", use_container_width=True, disabled=not st.session_state.api_ready or not st.session_state.tasks)
    brk_task_id = st.sidebar.selectbox("Break down task", ["- Select -"] + [f'{t["title"]} | {t["id"][:8]}' for t in st.session_state.tasks])
    brk_clicked = st.sidebar.button("Break down", use_container_width=True, disabled=not st.session_state.api_ready or brk_task_id == "- Select -")
    plan_hours = st.sidebar.slider("Today's available hours", min_value=1, max_value=12, value=6)
    plan_clicked = st.sidebar.button("Build daily plan", use_container_width=True, disabled=not st.session_state.api_ready or not st.session_state.tasks)

    if gen_clicked:
        if goal.strip():
            with st.spinner("Generating tasks..."):
                tasks = ai_generate_tasks_from_goal(goal.strip())
            for t in tasks:
                add_task_to_state(t)
            if tasks:
                st.sidebar.success(f"Added {len(tasks)} tasks.")
        else:
            st.sidebar.warning("Please provide a goal.")

    if pri_clicked:
        with st.spinner("Prioritizing tasks..."):
            st.session_state.tasks = ai_prioritize_tasks(st.session_state.tasks)
        st.sidebar.success("Tasks reprioritized.")

    if brk_clicked and brk_task_id != "- Select -":
        # Extract id from selection
        selected_id = brk_task_id.split("|")[-1].strip()
        task = get_task_by_id(selected_id)
        if task:
            with st.spinner("Breaking down task..."):
                subs = ai_breakdown_task(task)
            if subs:
                task["subtasks"] = subs
                st.sidebar.success(f"Added {len(subs)} subtasks.")
        else:
            st.sidebar.warning("Task not found.")

    if plan_clicked:
        with st.spinner("Building today's plan..."):
            plan = ai_daily_plan(st.session_state.tasks, plan_hours)
        if plan:
            st.session_state["today_plan"] = plan
            st.sidebar.success("Plan ready (see below).")
        else:
            st.sidebar.warning("Could not build plan.")

def toolbar():
    st.subheader("Controls")
    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1, 1])
    st.session_state.search = c1.text_input("Search", value=st.session_state.search, placeholder="Filter by title or details")
    st.session_state.filter_status = c2.selectbox("Status", ["All", "Todo", "Done"], index=["All","Todo","Done"].index(st.session_state.filter_status))
    st.session_state.sort_by = c3.selectbox("Sort by", ["Priority", "Due date", "Status"], index=["Priority","Due date","Status"].index(st.session_state.sort_by))
    if c4.button("Clear Completed"):
        clear_completed()
        st.success("Completed tasks cleared.")

    # Export / Import
    ec1, ec2 = st.columns([1,1])
    export_data = json.dumps(st.session_state.tasks, indent=2)
    ec1.download_button("Export JSON", data=export_data, file_name="tasks.json", mime="application/json", use_container_width=True)
    uploaded = ec2.file_uploader("Import JSON", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, list):
                # basic validation
                for t in data:
                    if "id" not in t:
                        t["id"] = str(uuid.uuid4())
                    if "status" not in t:
                        t["status"] = "todo"
                    if "priority" not in t:
                        t["priority"] = "Medium"
                    if "subtasks" not in t:
                        t["subtasks"] = []
                st.session_state.tasks = data
                st.success(f"Imported {len(data)} tasks.")
            else:
                st.warning("Invalid JSON format. Expecting a list of tasks.")
        except Exception as e:
            st.error(f"Failed to import: {e}")

def filter_and_sort_tasks() -> List[Dict[str, Any]]:
    tasks = st.session_state.tasks
    # Filter by status
    if st.session_state.filter_status != "All":
        tasks = [t for t in tasks if t["status"] == st.session_state.filter_status.lower()]
    # Search
    q = st.session_state.search.strip().lower()
    if q:
        tasks = [t for t in tasks if q in t["title"].lower() or q in t.get("details","").lower()]
    # Sort
    tasks = sort_tasks(tasks, st.session_state.sort_by)
    return tasks

def render_task_row(task: Dict[str, Any]):
    border_color = "#10b981" if task["status"] == "done" else "#e5e7eb"
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([0.05, 0.45, 0.2, 0.2, 0.1])
        status_label = "✅" if task["status"] == "done" else "⬜"
        if c1.button(status_label, key=f"toggle-{task['id']}", help="Toggle done/undone"):
            toggle_task_status(task["id"])
            st.experimental_rerun()
        c2.markdown(f"<div style='border-left:4px solid {border_color}; padding-left:8px'><b>{task['title']}</b></div>", unsafe_allow_html=True)
        c3.write(task["priority"])
        due_str = task["due_date"] or "-"
        c4.write(due_str)
        if c5.button("✖", key=f"del-{task['id']}", help="Delete"):
            delete_task(task["id"])
            st.experimental_rerun()
        with st.expander("Details / Edit", expanded=False):
            c21, c22 = st.columns([0.6, 0.4])
            new_title = c21.text_input("Title", value=task["title"], key=f"ed-title-{task['id']}")
            new_details = st.text_area("Details", value=task.get("details",""), key=f"ed-det-{task['id']}")
            c23, c24, c25 = st.columns([0.3, 0.3, 0.4])
            new_priority = c23.selectbox("Priority", ["High","Medium","Low"], index=["High","Medium","Low"].index(task["priority"]), key=f"ed-pri-{task['id']}")
            try:
                cur_due = datetime.fromisoformat(task["due_date"]).date() if task["due_date"] else None
            except Exception:
                cur_due = None
            new_due = c24.date_input("Due", value=cur_due, format="YYYY-MM-DD", key=f"ed-due-{task['id']}")
            if c25.button("Save", key=f"save-{task['id']}"):
                update_task(task["id"], new_title, new_details, new_priority, new_due if isinstance(new_due, date) else None)
                st.success("Task updated.")
                st.experimental_rerun()
            # Subtasks
            st.markdown("Subtasks")
            if task.get("subtasks"):
                for idx, sub in enumerate(task["subtasks"]):
                    sc1, sc2, sc3, sc4 = st.columns([0.05, 0.55, 0.25, 0.15])
                    if sc1.button("✅" if sub.get("status") == "done" else "⬜", key=f"subtoggle-{task['id']}-{idx}"):
                        task["subtasks"][idx]["status"] = "done" if sub.get("status") != "done" else "todo"
                        st.experimental_rerun()
                    sc2.text_input("Subtask", value=sub.get("title",""), key=f"sub-title-{task['id']}-{idx}", disabled=True)
                    sc3.text_input("Details", value=sub.get("details",""), key=f"sub-det-{task['id']}-{idx}", disabled=True)
                    if sc4.button("Remove", key=f"subrem-{task['id']}-{idx}"):
                        del task["subtasks"][idx]
                        st.experimental_rerun()
            else:
                st.info("No subtasks. Use 'Break down' in the sidebar to generate subtasks with AI.")

def render_tasks_list():
    tasks = filter_and_sort_tasks()
    if not tasks:
        st.info("No tasks to show.")
        return
    header = st.columns([0.05, 0.45, 0.2, 0.2, 0.1])
    header[0].markdown(" ")
    header[1].markdown("Task")
    header[2].markdown("Priority")
    header[3].markdown("Due")
    header[4].markdown(" ")
    st.divider()
    for task in tasks:
        render_task_row(task)

def render_today_plan():
    plan = st.session_state.get("today_plan")
    if not plan:
        return
    st.subheader("Today's Plan")
    tp = plan.get("today_plan", [])
    if tp:
        total = 0
        for item in tp:
            t1, t2, t3 = st.columns([0.5, 0.35, 0.15])
            t1.write(item.get("title",""))
            t2.write(item.get("reason",""))
            mins = int(item.get("time_estimate_minutes", 0) or 0)
            total += mins
            t3.write(f"{mins} min")
        st.write(f"Total: {total} min")
    notes = plan.get("notes")
    if notes:
        st.write("Notes:")
        st.write(notes)

# ----------------------------
# Main App
# ----------------------------

def main():
    st.set_page_config(page_title="AI To-Do List Agent", page_icon="✅", layout="wide")
    ensure_state()

    st.title("AI Task To-Do List Agent")
    st.caption("Plan, generate, prioritize, and track tasks. AI features require OPENAI_API_KEY environment variable.")

    sidebar_add_task()
    sidebar_ai_tools()
    toolbar()
    render_tasks_list()
    render_today_plan()

if __name__ == "__main__":
    main()