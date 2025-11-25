import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import io

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="DIFOT Converter", layout="wide")

DIFOT_COLUMNS: List[str] = [
    "EOW", "Month", "Week", "Agency", "Client", "Address",
    "Origin (WHO)", "Issue Area", "Category - Issue", "Subcategory",
    "Type", "Description (Hubspot)", "Notes (Hubspot)", "Ticket ID",
]

FEEDBACK_NATURE_TO_ISSUE = {
    "customer preference": ("Request", "Client Request"),
    "fsr defect": ("Client Defect", "Client Facing Defect"),
    "positive feedback": ("Other", "Other"),
}

DELIVERY_FEEDBACK_TO_SUBCATEGORY = {
    "extracted": "Extracted Images",
    "add ons": "Addons",
    "floorplan": "Floorplan",
    "virtual tour": "Virtual Tour",
    "pro images": "PRO Images",
    "on site": "Onsite",
}

# ==========================================
# HELPER FUNCTIONS (LOGIC PRESERVED)
# ==========================================

def _parse_over_by_hours(over_by: Optional[str]) -> Optional[float]:
    if not isinstance(over_by, str) or not over_by.strip():
        return None
    over_by = over_by.strip().lower()
    hours = 0
    minutes = 0
    match_h = re.search(r"(\d+)h", over_by)
    if match_h:
        try:
            hours = int(match_h.group(1))
        except ValueError:
            hours = 0
    match_m = re.search(r"(\d+)m", over_by)
    if match_m:
        try:
            minutes = int(match_m.group(1))
        except ValueError:
            minutes = 0
    if not match_h and match_m and minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    total = hours + minutes / 60.0
    return total

def _normalize_address(address: Optional[str]) -> Optional[str]:
    if not isinstance(address, str):
        return address
    first = address.split(";")[0].strip()
    parts = [p.strip() for p in first.split(" - ")]
    
    def looks_like_address(part: str) -> bool:
        if "," in part and any(ch.isdigit() for ch in part):
            return True
        if re.search(r"\b\d{4}\b", part):
            return True
        keywords = ["street", "road", "drive", "avenue", "court", "crescent", "place", "unit", "flat"]
        lower = part.lower()
        if any(word in lower for word in keywords) and any(ch.isdigit() for ch in part):
            return True
        return False
    
    candidates = [p for p in parts if looks_like_address(p)]
    if candidates:
        candidate = max(candidates, key=len)
    else:
        if len(parts) >= 3:
            candidate = parts[1]
        elif len(parts) == 2:
            p0, p1 = parts
            if looks_like_address(p1):
                candidate = p1
            elif looks_like_address(p0):
                candidate = p0
            else:
                candidate = p1
        else:
            candidate = parts[0]
    return candidate.strip()

def _flatten_multiline(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    text = value.replace("\u2022", "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return " | ".join(parts)

def _classify_hubspot_issue_area(feedback_nature: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(feedback_nature, str):
        key = feedback_nature.strip().lower()
        return FEEDBACK_NATURE_TO_ISSUE.get(key, (None, None))
    return (None, None)

def _map_hubspot_subcategory(delivery_feedback: Optional[str]) -> Optional[str]:
    if not isinstance(delivery_feedback, str):
        return None
    key = delivery_feedback.strip().lower()
    return DELIVERY_FEEDBACK_TO_SUBCATEGORY.get(key, delivery_feedback)

def _extract_client_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    return value.split(" (", 1)[0].strip()

def _compute_month_week_eow_from_series(date_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    created = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    month = created.dt.month_name().fillna("")
    week_num = ((created.dt.day - 1) // 7 + 1).astype("Int64")
    week = week_num.map(lambda x: f"Week {int(x)}" if pd.notna(x) else "")
    weekday = created.dt.weekday
    delta = (4 - weekday) % 7
    eow = created + pd.to_timedelta(delta, unit="D")
    eow_str = eow.dt.strftime('%Y-%m-%d').fillna("")
    return eow_str, month.astype("string"), week.astype("string")

def _clean_bracket_label(val: Optional[str]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    v = val.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1].strip()
    return v or None

def _classify_issue_area_from_category(cat: Optional[str]) -> Optional[str]:
    if not isinstance(cat, str):
        return None
    s = cat.lower()
    if "improvement" in s:
        return "Improvement"
    if "defect" in s or "late delivery" in s:
        return "Defect"
    return None

def _origin_from_category(cat: Optional[str]) -> Optional[str]:
    if not isinstance(cat, str):
        return None
    s = cat.upper()
    if "VTT" in s:
        return "VTT"
    if "QA" in s:
        return "QA"
    if "C3D" in s:
        return "C3D"
    return None

def _parse_address_from_task_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return name
    parts = [p.strip() for p in name.split(" - ")]
    if len(parts) >= 3:
        return parts[1]
    return name.strip()

def _parse_delivered_datetime(date_str: Optional[str]) -> Optional[datetime]:
    if not isinstance(date_str, str):
        return None
    s = date_str.strip()
    if not s:
        return None
    current_year = datetime.now().year
    fmt = "%d %b, %I:%M %p %Y"
    try:
        return datetime.strptime(f"{s} {current_year}", fmt)
    except Exception:
        return None

def _extract_address_from_job(job: Optional[str]) -> Optional[str]:
    if not isinstance(job, str):
        return job
    parts = [p.strip() for p in job.split(" - ")]
    if len(parts) >= 3:
        return parts[1]
    return job.strip()

def _parse_late_table(text: str) -> List[Tuple[str, str, str, str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    header_candidates = lines[0].lower().replace("  ", " ")
    if "job" in header_candidates and ("client" in header_candidates or "agency" in header_candidates):
        lines.pop(0)

    rows: List[Tuple[str, str, str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        parts_tab = re.split(r"\t+", line)
        if len(parts_tab) >= 5:
            parts = parts_tab
            job = parts[0].strip()
            client = parts[1].strip() if len(parts) > 1 else ""
            agency = parts[2].strip() if len(parts) > 2 else ""
            delivered = parts[3].strip() if len(parts) > 3 else ""
            over_by = parts[4].strip() if len(parts) > 4 else ""
            rows.append((job, client, agency, delivered, over_by))
            i += 1
            continue
        
        job_desc = line
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        remainder = re.split(r"\t+", next_line)
        
        if len(remainder) == 5:
            _, client, agency, delivered, over_by = [p.strip() for p in remainder]
        elif len(remainder) == 4:
            client, agency, delivered, over_by = [p.strip() for p in remainder]
        else:
            # Skip malformed lines instead of crashing the whole app
            i += 1
            continue
            
        rows.append((job_desc, client, agency, delivered, over_by))
        i += 2 
    return rows

# ==========================================
# CONVERSION LOGIC (ADAPTED FOR STREAMLIT)
# ==========================================

def convert_hubspot_file(uploaded_file) -> pd.DataFrame:
    hub = pd.read_csv(uploaded_file)
    eow, month, week = _compute_month_week_eow_from_series(hub.get("Create date"))
    
    issue_area_list = []
    cat_issue_list = []
    for val in hub.get("Feedback Nature", []):
        ia, ci = _classify_hubspot_issue_area(val)
        issue_area_list.append(ia)
        cat_issue_list.append(ci)

    subcategories = hub.get("Delivery Feedback").map(_map_hubspot_subcategory)
    addresses = hub.get("Associated Deal").copy()
    mask_missing = addresses.isna() | addresses.eq("")

    def _parse_ticket_for_address(ticket: Optional[str]) -> Optional[str]:
        if not isinstance(ticket, str):
            return ticket
        m = re.search(r"re:\s*(.+)", ticket, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            candidate_parts = candidate.split(" - ")
            return candidate_parts[0].strip()
        parts = [p.strip() for p in ticket.split(" - ")]
        for part in parts:
            if any(ch.isdigit() for ch in part) and "," in part:
                return part
        return ticket.strip()

    if mask_missing.any():
        fallback_addresses = hub.loc[mask_missing, "Ticket name"].apply(_parse_ticket_for_address)
        addresses.loc[mask_missing] = fallback_addresses

    df_out = pd.DataFrame(columns=DIFOT_COLUMNS)
    df_out["EOW"] = eow
    df_out["Month"] = month
    df_out["Week"] = week
    df_out["Agency"] = hub.get("Associated Company (Primary)")
    df_out["Client"] = hub.get("Associated Contact").map(_extract_client_name)
    df_out["Address"] = addresses.map(_normalize_address)
    df_out["Origin (WHO)"] = "Client"
    df_out["Issue Area"] = issue_area_list
    df_out["Category - Issue"] = cat_issue_list
    df_out["Subcategory"] = subcategories
    df_out["Description (Hubspot)"] = hub.get("What was the feedback?")
    df_out["Notes (Hubspot)"] = hub.get("Additional Information")
    
    ticket_id_series = None
    if "Ticket ID" in hub.columns:
        ticket_id_series = hub.get("Ticket ID")
    elif "Ticket ID.1" in hub.columns:
        ticket_id_series = hub.get("Ticket ID.1")
    df_out["Ticket ID"] = ticket_id_series

    df_out["Description (Hubspot)"] = df_out["Description (Hubspot)"].map(_flatten_multiline)
    df_out["Notes (Hubspot)"] = df_out["Notes (Hubspot)"].map(_flatten_multiline)
    
    # Fill remaining required columns with None/Empty
    for col in DIFOT_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = None
            
    return df_out[DIFOT_COLUMNS]

def convert_clickup_file(uploaded_file) -> pd.DataFrame:
    click = pd.read_csv(uploaded_file)
    due_series = pd.to_datetime(click.get("Due Date"), errors="coerce", dayfirst=True)
    
    eow_list, month_list, week_list = [], [], []
    for dt in due_series:
        if pd.isna(dt):
            eow_list.append("")
            month_list.append("")
            week_list.append("")
        else:
            py_dt = dt.to_pydatetime()
            month_list.append(py_dt.strftime("%B"))
            week_num = (py_dt.day - 1) // 7 + 1
            week_list.append(f"Week {week_num}")
            weekday = py_dt.weekday()
            delta_days = (4 - weekday) % 7
            eow_dt = py_dt + timedelta(days=delta_days)
            eow_list.append(eow_dt.strftime("%Y-%m-%d"))

    category_clean = click.get("Issue - Category (labels)").map(_clean_bracket_label)
    subcategory_clean = click.get("Issue - Subcategory (labels)").map(_clean_bracket_label)
    issue_area = category_clean.map(_classify_issue_area_from_category)
    origin = category_clean.map(_origin_from_category)
    address = click.get("Task Name").map(_parse_address_from_task_name)

    df_out = pd.DataFrame(columns=DIFOT_COLUMNS)
    df_out["EOW"] = eow_list
    df_out["Month"] = month_list
    df_out["Week"] = week_list
    df_out["Agency"] = click.get("Agency (short text)")
    df_out["Client"] = click.get("Booking Agent Name (short text)")
    df_out["Address"] = address.map(_normalize_address)
    df_out["Origin (WHO)"] = origin
    df_out["Issue Area"] = issue_area
    df_out["Category - Issue"] = category_clean
    df_out["Subcategory"] = subcategory_clean
    df_out["Notes (Hubspot)"] = click.get("Notes (Hubspot)", pd.Series([None]*len(click))) # Handle missing logic if column exists or not
    df_out["Notes (Hubspot)"] = df_out["Notes (Hubspot)"].map(_flatten_multiline)

    for col in DIFOT_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = None

    return df_out[DIFOT_COLUMNS]

def convert_late_text_to_df(text: str) -> pd.DataFrame:
    parsed_rows = _parse_late_table(text)
    data = {col: [] for col in DIFOT_COLUMNS}

    for job, client, agency, delivered_str, over_by_str in parsed_rows:
        dt = _parse_delivered_datetime(delivered_str)
        if dt is None:
            eow_str, month_name, week_str = "", "", ""
        else:
            month_name = dt.strftime("%B")
            week_num = (dt.day - 1) // 7 + 1
            week_str = f"Week {week_num}"
            weekday = dt.weekday()
            delta_days = (4 - weekday) % 7
            eow_dt = dt + timedelta(days=delta_days)
            eow_str = eow_dt.strftime("%Y-%m-%d")

        address = _normalize_address(_extract_address_from_job(job))
        
        notes_parts = []
        if delivered_str: notes_parts.append(f"Delivered: {delivered_str}")
        if over_by_str: notes_parts.append(f"Over by: {over_by_str}")
        notes = " | ".join(notes_parts)

        hours_late = _parse_over_by_hours(over_by_str)
        if hours_late is not None:
            if hours_late < 5:
                issue_area_val, category_val = "Improvement", "Late Delivery (After 12PM)"
            else:
                issue_area_val, category_val = "Defect", "Late Delivery (After 5PM)"
        else:
            issue_area_val, category_val = "Defect", "Late Delivery"

        data["EOW"].append(eow_str)
        data["Month"].append(month_name)
        data["Week"].append(week_str)
        data["Agency"].append(agency)
        data["Client"].append(client)
        data["Address"].append(address)
        data["Origin (WHO)"].append("Late Delivery")
        data["Issue Area"].append(issue_area_val)
        data["Category - Issue"].append(category_val)
        data["Subcategory"].append("")
        data["Type"].append("")
        data["Description (Hubspot)"].append("")
        data["Notes (Hubspot)"].append(notes)
        data["Ticket ID"].append("")

    df = pd.DataFrame(data, columns=DIFOT_COLUMNS)
    if not df.empty:
        df["Notes (Hubspot)"] = df["Notes (Hubspot)"].map(_flatten_multiline)
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False, header=False).encode('utf-8')

def convert_df_to_tsv_string(df, include_header=True):
    if df is None or df.empty:
        return ""
    df_clean = df.fillna("")
    df_clean = df_clean.applymap(lambda val: _flatten_multiline(val) if isinstance(val, str) else val)
    
    # We use io.StringIO to let pandas handle the TSV generation cleanly
    output = io.StringIO()
    df_clean.to_csv(output, sep='\t', index=False, header=include_header)
    return output.getvalue()


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("DIFOT Converter Web App")
st.markdown("Convert HubSpot, ClickUp, and Late Delivery data into the unified DIFOT layout.")

# Initialize Session State
if 'hubspot_df' not in st.session_state:
    st.session_state['hubspot_df'] = pd.DataFrame()
if 'clickup_df' not in st.session_state:
    st.session_state['clickup_df'] = pd.DataFrame()
if 'late_df' not in st.session_state:
    st.session_state['late_df'] = pd.DataFrame()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["HubSpot", "ClickUp", "Late Delivery", "Combine All"])

# --- TAB 1: HUBSPOT ---
with tab1:
    st.header("HubSpot to DIFOT")
    uploaded_hs = st.file_uploader("Upload HubSpot CSV", type=['csv'])
    
    if uploaded_hs is not None:
        if st.button("Convert HubSpot"):
            try:
                st.session_state['hubspot_df'] = convert_hubspot_file(uploaded_hs)
                st.success(f"Converted {len(st.session_state['hubspot_df'])} rows.")
            except Exception as e:
                st.error(f"Error converting file: {e}")

    if not st.session_state['hubspot_df'].empty:
        st.dataframe(st.session_state['hubspot_df'].head())
        st.download_button(
            "Download HubSpot CSV",
            data=convert_df_to_csv(st.session_state['hubspot_df']),
            file_name="hubspot_difot.csv",
            mime="text/csv"
        )

# --- TAB 2: CLICKUP ---
with tab2:
    st.header("ClickUp to DIFOT")
    uploaded_cu = st.file_uploader("Upload ClickUp CSV", type=['csv'])
    
    if uploaded_cu is not None:
        if st.button("Convert ClickUp"):
            try:
                st.session_state['clickup_df'] = convert_clickup_file(uploaded_cu)
                st.success(f"Converted {len(st.session_state['clickup_df'])} rows.")
            except Exception as e:
                st.error(f"Error converting file: {e}")

    if not st.session_state['clickup_df'].empty:
        st.dataframe(st.session_state['clickup_df'].head())
        st.download_button(
            "Download ClickUp CSV",
            data=convert_df_to_csv(st.session_state['clickup_df']),
            file_name="clickup_difot.csv",
            mime="text/csv"
        )

# --- TAB 3: LATE DELIVERY ---
with tab3:
    st.header("Late Delivery to DIFOT")
    late_input = st.text_area("Paste table here (Job / Client / Agency / Delivered / Over by)", height=200)
    
    if st.button("Convert Late Delivery"):
        if late_input.strip():
            try:
                st.session_state['late_df'] = convert_late_text_to_df(late_input)
                st.success(f"Converted {len(st.session_state['late_df'])} rows.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please paste data first.")

    if not st.session_state['late_df'].empty:
        st.dataframe(st.session_state['late_df'].head())
        
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.download_button(
                "Download CSV",
                data=convert_df_to_csv(st.session_state['late_df']),
                file_name="late_difot.csv",
                mime="text/csv"
            )
        with col_l2:
            # For copying, we display a code block with the TSV
            st.text("Copy this content for Excel/Sheets:")
            tsv_data = convert_df_to_tsv_string(st.session_state['late_df'])
            st.code(tsv_data, language="text")

# --- TAB 4: COMBINE ---
with tab4:
    st.header("Combine All Data")
    
    dfs = []
    if not st.session_state['hubspot_df'].empty: dfs.append(st.session_state['hubspot_df'])
    if not st.session_state['clickup_df'].empty: dfs.append(st.session_state['clickup_df'])
    if not st.session_state['late_df'].empty: dfs.append(st.session_state['late_df'])
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Remove header rows if accidentally included in data
        combined = combined[combined["EOW"].astype(str).str.lower() != "eow"]
        # Filter blank rows
        tmp = combined.fillna("")
        mask_blank = tmp.apply(lambda row: all(str(val).strip() == "" for val in row), axis=1)
        combined = combined.loc[~mask_blank]
        
        # Standardize dates
        combined['EOW'] = pd.to_datetime(combined['EOW'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d').fillna(combined['EOW'])
        
        st.info(f"Total Combined Rows: {len(combined)}")
        st.dataframe(combined.head())
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.download_button(
                "Download Combined CSV (No Header)",
                data=convert_df_to_csv(combined),
                file_name="combined_difot.csv",
                mime="text/csv"
            )
        with col_c2:
            st.text("Copy for Excel/Sheets (TSV):")
            # TSV generation without header for copy paste
            tsv_combined = convert_df_to_tsv_string(combined, include_header=False)
            st.code(tsv_combined, language="text")
            
    else:
        st.warning("No data converted yet. Go to previous tabs to convert files.")
