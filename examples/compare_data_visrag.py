
import pandas as pd
import polars as pl

def load_parquet(root=None, return_type='pd'):
    """
    Load data from Parquet files with optional filtering on date_id, time_id, and selected columns.

    Parameters:
    - date_id_range (tuple, optional): Range of date_id to filter (start, end). Default is None, which means all dates.
    - time_id_range (tuple, optional): Range of time_id to filter (start, end). Default is None, which means all times.
    - columns (list, optional): List of columns to load. Default is None, which means all columns.
    - return_type (str, optional): Type of data to return ('pl' for Polars DataFrame or 'pd' for Pandas DataFrame). Default is 'pl'.

    Returns:a
    - pl.DataFrame or pd.DataFrame: The filtered data as a Polars or Pandas DataFrame.
    """
    # Load data using Polars lazy loading (scan_parquet)
    data = pl.scan_parquet(f"{root}")

    # Collect the data to execute the lazy operations
    if return_type == 'pd':
        return data.collect().to_pandas()
    else:
        return data.collect()

if __name__ == "__main__":

    datasets = [ ### lmms-eval与visRAG重叠的数据集
        # "ArxivQA",
        "ChartQA",
        # "MP-DocVQA",
        "InfoVQA",
        # "PlotQA",
        # "SlideVQA",
    ]

    ### datasource of visRAG
    root1 = '/ssddata/liuyue/github/VisRAG/qa_datasets'
    file_temp1 = "VisRAG-Ret-Test-{}"



    ### datasource of lmms-eval
    root2 = '/ssddata/liuyue/github/LLaVA-NeXT/qa_datasets'

    for dataset in datasets:
        print(f'Dataset: {dataset}')

        corpus_df = load_parquet(f'{root1}/{file_temp1.format(dataset)}/corpus/*.parquet')
        qrels_df = load_parquet(f'{root1}/{file_temp1.format(dataset)}/qrels/*.parquet')
        query_df = load_parquet(f'{root1}/{file_temp1.format(dataset)}/queries/*.parquet')

        print(list(corpus_df.columns))
        print(list(qrels_df.columns))
        print(list(query_df.columns))

        query_df = query_df[['query', 'answer']]
        print(f'Query_visRAG: {query_df.shape}')
        # query_df.loc[0, 'query'] = 'ababa' ## test whether isin works as expect

        if dataset == 'ChartQA':
            df2 = load_parquet(f'{root2}/ChartQA/data/*.parquet')
            df2 = df2[['question', 'answer']]
        elif dataset == 'InfoVQA':
            df2 = load_parquet(f'{root2}/DocVQA/InfographicVQA/valid*.parquet')
            df2 = df2[['question', 'answers']]

        print(list(df2.columns))
        print(f'query_lmms-eval: {df2.shape}')
        
        se3 = query_df['query'].isin(df2.question).astype(int)
        query_df = query_df.assign(indf2=se3)

        print(f'visRAG {se3.sum()}/{query_df.shape[0]} queries in df2 queries {df2.shape[0]}')

        print('='*30)
    pass