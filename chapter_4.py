from io import StringIO

import pandas as pd

if __name__ == '__main__':
    csv_data = """
        A,B,C,D
        1.0,2.0,3.0,4.0
        5.0,6.0,,8.0
        10.0,11.0,12.0,
    """
    df = pd.read_csv(StringIO(csv_data))
    print(df, end='\n\n')
    print(df.isnull().sum(), end='\n\n')
    print(df.values, end='\n\n')
    print(df.dropna(), end='\n\n')
    print(df.dropna(axis=1), end='\n\n')
    print(df.dropna(how='all'), end='\n\n')
    print(df.dropna(thresh=4), end='\n\n')
    print(df.dropna(subset=['C']), end='\n\n')
