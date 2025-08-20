import streamlit as st
import pandas as pd
import os
from typing import List

try:
    import duckdb
except Exception:
    duckdb = None

DATA_DIR = "data"


def list_data_files(data_dir: str = DATA_DIR) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".tsv", ".xlsx"))]


@st.cache_data
def load_dataframe_from_path(path: str) -> pd.DataFrame:
    if path.lower().endswith('.xlsx'):
        return pd.read_excel(path)
    if path.lower().endswith('.tsv'):
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


@st.cache_data
def load_dataframe_from_buffer(buf) -> pd.DataFrame:
    # buf is a BytesIO / UploadedFile
    name = getattr(buf, "name", "upload.csv")
    if name.lower().endswith('.xlsx'):
        return pd.read_excel(buf)
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, encoding='utf-8', errors='replace')


def safe_sql_check(sql: str) -> bool:
    # Very simple safety: only allow one SELECT, block dangerous keywords
    s = sql.lower()
    forbidden = ['insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'attach ', ';']
    return not any(k in s for k in forbidden)


def build_sql(selected_cols: List[str], aggs: List[dict], group_by: List[str], filters: List[dict], table_name: str = 'df') -> str:
    if aggs:
        agg_parts = []
        for a in aggs:
            func = a.get('func')
            col = a.get('col')
            alias = a.get('alias') or f"{func}_{col}"
            agg_parts.append(f"{func.upper()}({col}) AS {alias}")
        select_clause = ', '.join(group_by + agg_parts) if group_by else ', '.join(agg_parts)
    else:
        if selected_cols:
            select_clause = ', '.join(selected_cols)
        else:
            select_clause = '*'

    where_clause = ''
    if filters:
        conds = []
        for f in filters:
            col = f['col']
            op = f['op']
            val = f['val']
            # simple numeric detection could be improved
            if val is None or val == '':
                continue
            if isinstance(val, (int, float)) or val.replace('.', '', 1).isdigit():
                conds.append(f"{col} {op} {val}")
            else:
                # quote single quotes inside
                v = val.replace("'", "''")
                conds.append(f"{col} {op} '{v}'")
        if conds:
            where_clause = 'WHERE ' + ' AND '.join(conds)

    group_clause = ''
    if group_by:
        group_clause = 'GROUP BY ' + ', '.join(group_by)

    sql = f"SELECT {select_clause} FROM {table_name} {where_clause} {group_clause};"
    return sql


def main():
    st.title('Drag-like SQL Builder')
    st.markdown('A simple drag-like interface to build SELECT queries and run them (DuckDB recommended).')

    source = st.radio('Data source', ['Server file', 'Upload CSV/XLSX'], index=0)

    df = None
    file_path = None
    if source == 'Server file':
        files = list_data_files()
        if not files:
            st.warning('No files found in `data/`. Upload a CSV or place files into the project `data/` folder.')
        else:
            sel = st.selectbox('Choose file', files)
            if sel:
                file_path = os.path.join(DATA_DIR, sel)
                try:
                    df = load_dataframe_from_path(file_path)
                except Exception as e:
                    st.error(f'Failed to read {sel}: {e}')
    else:
        uploaded = st.file_uploader('Upload CSV or XLSX', type=['csv', 'tsv', 'xlsx'])
        if uploaded is not None:
            try:
                df = load_dataframe_from_buffer(uploaded)
            except Exception as e:
                st.error(f'Failed to read uploaded file: {e}')

    if df is None:
        st.stop()

    st.sidebar.header('Table preview & schema')
    st.sidebar.write(f'Rows: {len(df):,} | Columns: {len(df.columns)}')
    st.sidebar.dataframe(df.head(5))

    cols = list(df.columns.astype(str))

    if 'selected_cols' not in st.session_state:
        st.session_state.selected_cols = []
    if 'filters' not in st.session_state:
        st.session_state.filters = []
    if 'aggs' not in st.session_state:
        st.session_state.aggs = []
    if 'group_by' not in st.session_state:
        st.session_state.group_by = []

    st.subheader('Columns')
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('**Available**')
        for c in cols:
            if c not in st.session_state.selected_cols:
                if st.button(f'+ {c}', key=f'add_{c}'):
                    st.session_state.selected_cols.append(c)
                    st.experimental_rerun()
    with c2:
        st.markdown('**Selected (order matters)**')
        for c in list(st.session_state.selected_cols):
            st.write(c, end='')
            if st.button(f'− {c}', key=f'rem_{c}'):
                st.session_state.selected_cols.remove(c)
                st.experimental_rerun()

    st.markdown('---')
    st.subheader('Filters')
    with st.form('filter_form', clear_on_submit=True):
        fcol = st.selectbox('Column', cols)
        fop = st.selectbox('Operator', ['=', '!=', '>', '<', '>=', '<=', 'LIKE'])
        fval = st.text_input('Value (no quotes)')
        if st.form_submit_button('Add filter'):
            st.session_state.filters.append({'col': fcol, 'op': fop, 'val': fval})

    if st.session_state.filters:
        st.write('Current filters:')
        for i, f in enumerate(st.session_state.filters):
            st.write(f"{i+1}. {f['col']} {f['op']} {f['val']}")
            if st.button(f'Remove filter {i}', key=f'rem_filter_{i}'):
                st.session_state.filters.pop(i)
                st.experimental_rerun()

    st.markdown('---')
    st.subheader('Aggregations')
    with st.form('agg_form', clear_on_submit=True):
        acol = st.selectbox('Aggregate column', cols)
        afunc = st.selectbox('Function', ['sum', 'avg', 'count', 'min', 'max'])
        aalias = st.text_input('Alias (optional)')
        if st.form_submit_button('Add aggregation'):
            st.session_state.aggs.append({'col': acol, 'func': afunc, 'alias': aalias})

    if st.session_state.aggs:
        st.write('Aggregations:')
        for i, a in enumerate(st.session_state.aggs):
            st.write(f"{i+1}. {a['func'].upper()}({a['col']}) AS {a.get('alias') or a['func']+'_'+a['col']}")
            if st.button(f'Remove agg {i}', key=f'rem_agg_{i}'):
                st.session_state.aggs.pop(i)
                st.experimental_rerun()

    st.markdown('---')
    st.subheader('Group by')
    gb = st.multiselect('Group by columns', cols, default=st.session_state.group_by)
    st.session_state.group_by = gb

    st.markdown('---')
    st.subheader('Generate & Run')
    sql = build_sql(st.session_state.selected_cols, st.session_state.aggs, st.session_state.group_by, st.session_state.filters)
    st.code(sql)

    if not safe_sql_check(sql):
        st.error('Generated SQL failed safety check. Aborting execution.')
    else:
        if st.button('Execute SQL'):
            with st.spinner('Running query...'):
                try:
                    if duckdb is None:
                        st.error('duckdb is not installed in the environment. Please install duckdb to run SQL queries (pip install duckdb).')
                    else:
                        con = duckdb.connect(':memory:')
                        con.register('df', df)
                        try:
                            res = con.execute(sql).df()
                        finally:
                            con.unregister('df')
                            con.close()

                        st.success(f'Returned {len(res):,} rows')
                        st.dataframe(res)

                        # simple auto chart: if group_by present and one numeric agg, draw bar
                        if st.session_state.group_by and st.session_state.aggs:
                            # pick first agg that isn't count
                            a = st.session_state.aggs[0]
                            if a['func'] in ('sum', 'avg', 'min', 'max'):
                                x = st.session_state.group_by[0]
                                y = a.get('alias') or f"{a['func']}_{a['col']}"
                                if y in res.columns:
                                    st.bar_chart(res.set_index(x)[y])

                        # download
                        csv = res.to_csv(index=False).encode('utf-8')
                        st.download_button('Download CSV', csv, file_name='query_result.csv', mime='text/csv')
                except Exception as e:
                    st.error(f'Query failed: {e}')


if __name__ == '__main__':
    main()
