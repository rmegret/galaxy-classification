def move_columns_to_front(df, columns_to_move):
  """
  Move specific columns to the front of a DataFrame.

  Parameters:
  df (pd.DataFrame): The DataFrame to modify.
  columns_to_move (list): List of column names to move to the front.

  Returns:
  pd.DataFrame: The modified DataFrame with specified columns at the front.
  """
  return df[columns_to_move + [col for col in df.columns if col not in columns_to_move]]
