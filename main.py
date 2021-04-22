import pandas as pd
import numpy as np


def read_raw_data(format_dates=True):
    """
    This function reads and returns the raw data sets invoice and line_item.
    :return:
    """
    # Read raw data sets
    invoice = pd.read_csv("raw_data/invoice.csv", low_memory=False)
    line_item = pd.read_csv("raw_data/line_item.csv", low_memory=False)

    # Format dates
    if format_dates:
        # Invoice
        for d in ['invoice_date', 'due_date', 'period_start_date', 'period_end_date']:
            invoice[d] = pd.to_datetime(invoice[d], format='%Y-%m-%d', errors='coerce')
        # Line item
        for d in ['period_start_date', 'period_end_date']:
            line_item[d] = pd.to_datetime(line_item[d], format='%Y-%m-%d', errors='coerce')

    return invoice, line_item


def vendor_not_seen_in_a_while(invoice):
    """
    This function obtains the vendor not seen in a while glean.
    :param pd.DataFrame invoice: invoice data.
    :return: pd.DataFrame of vendor not seen in a while glean.
    """
    # Copy to not modify input df
    df = invoice.copy()

    # Get differences of invoice dates in days
    df.sort_values(by=['canonical_vendor_id', 'invoice_date'], ascending=True, inplace=True)
    df['vendor_not_seen_in_a_while'] = \
        df.groupby(by='canonical_vendor_id')['invoice_date'].agg('diff').fillna(pd.Timedelta(days=-1))

    # Generate vendor not seen in a while glean text
    df['glean_text'] = \
        df.apply(
            lambda row: f"First new bill in {np.round(row['vendor_not_seen_in_a_while'].days / 30, 2)} months from vendor "
                        f"{row['canonical_vendor_id']}" if row['vendor_not_seen_in_a_while'].days > 90 else None,
            axis=1
        )
    # Glean date corresponds to the invoice date
    df['glean_date'] = df['invoice_date']
    # Glean type --> 1: vendor_not_seen_in_a_while
    df['glean_type'] = 1
    # Glean location --> 1: invoice
    df['glean_location'] = 1

    # Only return registers that triggered the glean
    df = df[~df['glean_text'].isna()]
    # Only return variables needed for the output
    columns_to_deliver = ['glean_date', 'glean_text', 'glean_type', 'glean_location', 'invoice_id',
                          'canonical_vendor_id']
    return df[columns_to_deliver]


def accrual_alert(invoice, line_item):
    """
    This function gets the accrual alert glean.
    :param pd.DataFrame invoice: invoice data.
    :param pd.DataFrame line_item: line item data.
    :return: pd.DataFrame of accrual alert glean.
    """
    # Merge invoice and line_item on invoice_id. Copy to not modify input df
    df = invoice.merge(right=line_item,
                       on='invoice_id',
                       how='left').copy()

    # If there are multiple end dates, pick the last one.
    df['period_end_date'] = df[['period_end_date_x', 'period_end_date_y']].max(axis=1)
    df.sort_values(by='period_end_date', ascending=True, inplace=True)
    end_dates = pd.DataFrame(df.groupby(by='invoice_id')['period_end_date'].agg('max')).rename(
        columns={'period_end_date': 'latest_period_end_date'})

    # Compare period end date with invoice date to generate the accrual alert glean text
    df = invoice.merge(right=end_dates, on='invoice_id')
    df['accrual_alert'] = df['latest_period_end_date'] - df['invoice_date']
    df['glean_text'] = \
        df.apply(
            lambda row: f"Line items from vendor {row['canonical_vendor_id']} in this invoice cover future periods "
                        f"(through {row['latest_period_end_date'].date()})" if row['accrual_alert'].days > 90 else None,
            axis=1
        )

    # Glean date corresponds to the invoice date
    df['glean_date'] = df['invoice_date']
    # Glean type --> 2: accrual_alert
    df['glean_type'] = 2
    # Glean location --> 1: invoice
    df['glean_location'] = 1

    # Only return registers that triggered the glean
    df = df[~df['glean_text'].isna()]
    # Only return variables needed for the output
    columns_to_deliver = ['glean_date', 'glean_text', 'glean_type', 'glean_location', 'invoice_id',
                          'canonical_vendor_id']
    return df[columns_to_deliver]


def large_month_increase_mtd(invoice):
    """
    This function performs the large month increase mtd glean.
    :param pd.DataFrame invoice: invoice data.
    :return: pd.DataFrame of large month increase mtd glean.
    """
    # Generate monthly date range (monthly start) for each vendor
    min_invoice_date = invoice['invoice_date'].min().replace(day=1)
    max_invoice_date = invoice['invoice_date'].max().replace(day=1) + pd.DateOffset(months=1)
    date_range = pd.date_range(start=min_invoice_date, end=max_invoice_date, freq='MS')
    df = pd.concat((pd.DataFrame({'invoice_date': date_range}).assign(canonical_vendor_id=vendor_id)
                    for vendor_id in invoice['canonical_vendor_id'].unique()),
                   ignore_index=True)

    # Copy to not modify input df
    df2 = invoice.copy()
    # Reset invoice date at the start of the month
    df2['invoice_date'] = df2['invoice_date'].apply(lambda x: x.replace(day=1))
    # Merge with df
    df = df.merge(right=df2[['invoice_id', 'invoice_date', 'total_amount', 'canonical_vendor_id']],
                  on=['canonical_vendor_id', 'invoice_date'],
                  how='left'
                  )
    # Get monthly spend
    df = df.groupby(by=['canonical_vendor_id', 'invoice_date'])['total_amount'].agg('sum').reset_index()
    # Get 12-month average spend
    df['twelve_month_avg'] = \
        df.groupby(by='canonical_vendor_id')['total_amount'].rolling(window=12).mean().reset_index()['total_amount']
    # Generate large month increase mtd glean text
    df['glean_text'] = df.apply(lmi_mtd, axis=1)

    # Glean date corresponds to the invoice date
    df['glean_date'] = df['invoice_date']
    # Glean type --> 3: large_month_increase_mtd
    df['glean_type'] = 3
    # Glean location --> 2: vendor
    df['glean_location'] = 2
    # No invoice id
    df['invoice_id'] = None

    # Only return registers that triggered the glean
    df = df[~df['glean_text'].isna()]
    # Only return variables needed for the output
    columns_to_deliver = ['glean_date', 'glean_text', 'glean_type', 'glean_location', 'invoice_id',
                          'canonical_vendor_id']
    return df[columns_to_deliver]


def lmi_mtd(row):
    """
    Apply this function to each row of the processed data set to obtain the corresponding text.
    :param pd.Series row: row of the large month increase mtd glean data frame.
    :return: str of the corresponding glean text.
    """
    # Conditions to trigger the large month increase mtd glean
    condition1 = (row['total_amount'] > 10000) and (row['total_amount'] > 0.5 * row['twelve_month_avg'])
    condition2 = (row['total_amount'] < 10000) and (row['total_amount'] > 1000) and \
                 (row['total_amount'] > 2 * row['twelve_month_avg'])
    condition3 = (row['total_amount'] < 1000) and (row['total_amount'] > 5 * row['twelve_month_avg'])
    condition4 = row['total_amount'] < 100

    # Text generation according to conditions
    if condition4:
        return None
    elif condition1 or condition2 or condition3:
        return f"Monthly spend with {row['canonical_vendor_id']} is ${row['total_amount']} " \
               f"({np.round(row['total_amount'] / row['twelve_month_avg'] * 100, 2)}%) higher than average."
    else:
        return None


def no_invoice_received_monthly(invoice):
    """
    This function performs the no invoice received glean (monthly case).
    :param pd.DataFrame invoice: invoice data.
    :return: pd.DataFrame of no invoice received glean (monthly case).
    """
    # Generate monthly date range (monthly start) for each vendor
    min_invoice_date = invoice['invoice_date'].min().replace(day=1)
    max_invoice_date = invoice['invoice_date'].max().replace(day=1) + pd.DateOffset(months=1)
    monthly_date_range = pd.date_range(start=min_invoice_date, end=max_invoice_date, freq='MS')
    df = pd.concat((pd.DataFrame({'invoice_date_reset': monthly_date_range}).assign(canonical_vendor_id=vendor_id)
                    for vendor_id in invoice['canonical_vendor_id'].unique()),
                   ignore_index=True)

    # Copy to not modify input df
    df2 = invoice.copy()
    # Reset invoice date at the start of the month
    df2['invoice_date_reset'] = df2['invoice_date'].apply(lambda x: x.replace(day=1))
    df2 = df2[['canonical_vendor_id', 'invoice_date', 'invoice_date_reset']]
    # Get the most frequent invoice day (wrt month)
    df2['invoice_date_day'] = df2['invoice_date'].dt.day
    most_frequent_day_dict = df2.groupby(by='canonical_vendor_id')['invoice_date_day']. \
        agg(lambda x: x.value_counts().index[0] if x.count() > 0 else 0).to_dict()
    df2.drop_duplicates(subset=['canonical_vendor_id', 'invoice_date'], inplace=True)
    # Keep only the first invoice date of the month
    df2 = df2.groupby(by=['canonical_vendor_id', 'invoice_date_reset'])['invoice_date'].agg('min').reset_index()
    # Flag of invoice generated in the corresponding month
    df2['invoice_bool'] = 1
    # Merge with df
    df = df.merge(right=df2,
                  on=['canonical_vendor_id', 'invoice_date_reset'],
                  how='left'
                  )
    df['invoice_bool'] = df['invoice_bool'].fillna(value=0)

    # Get the invoice flag for each month
    df = df.groupby(by=['canonical_vendor_id', 'invoice_date_reset'])['invoice_bool'].agg('max').reset_index()
    # If invoice generated for three consecutive months the following variable is 1
    df['consecutive_three_month'] = \
        df.groupby(by='canonical_vendor_id')['invoice_bool'].rolling(window=3).mean().reset_index()['invoice_bool']
    # To compare for the next month
    df['consecutive_three_month'] = df['consecutive_three_month'].shift(1)
    # Get real invoice dates
    df = df.merge(df2[['canonical_vendor_id', 'invoice_date_reset', 'invoice_date']],
                  on=['canonical_vendor_id', 'invoice_date_reset'],
                  how='left'
                  )

    # Generate daily date range for each vendor
    daily_date_range = pd.date_range(start=min_invoice_date, end=max_invoice_date, freq='D')
    daily_dates = pd.concat((pd.DataFrame({'daily_date': daily_date_range}).assign(canonical_vendor_id=vendor_id)
                             for vendor_id in invoice['canonical_vendor_id'].unique()),
                            ignore_index=True)
    # Merge with df on each month and vendor
    daily_dates['invoice_date_reset'] = daily_dates['daily_date'].apply(lambda x: x.replace(day=1))
    df = df.merge(right=daily_dates,
                  on=['canonical_vendor_id', 'invoice_date_reset'],
                  how='right'
                  )

    # Generate no invoice received glean (monthly case) text
    df['glean_text'] = df.apply(alarm_no_invoice_monthly,
                                args=(most_frequent_day_dict,),
                                axis=1)

    # Glean date corresponds to the daily date
    df['glean_date'] = df['daily_date']
    # Glean type --> 4: no_invoice_received
    df['glean_type'] = 4
    # Glean location --> 2: vendor
    df['glean_location'] = 2
    # No invoice id
    df['invoice_id'] = None

    # Only return registers that triggered the glean
    df = df[~df['glean_text'].isna()]
    # Only return variables needed for the output
    columns_to_deliver = ['glean_date', 'glean_text', 'glean_type', 'glean_location', 'invoice_id',
                          'canonical_vendor_id']
    return df[columns_to_deliver]


def alarm_no_invoice_monthly(row, most_frequent_day_dict):
    """
    Apply this function to each row of the processed data set to obtain the corresponding text.
    :param pd.Series row: row of the no invoice monthly glean data frame.
    :param dict most_frequent_day_dict: dictionary of most frequent day by vendor.
    :return: str of the corresponding glean text.
    """
    # Variables
    invoice_bool = row['invoice_bool']
    consecutive_three_month = row['consecutive_three_month']
    invoice_date = row['invoice_date']
    date = row['daily_date']
    vendor = row['canonical_vendor_id']

    # Conditions to trigger the no invoice monthly glean
    condition1 = invoice_bool == 0
    condition2 = consecutive_three_month == 1
    condition3 = date.day > most_frequent_day_dict[vendor]
    condition4 = invoice_bool == 1
    condition5 = (invoice_date.month == date.month) & (invoice_date.year == date.year)
    condition6 = invoice_date.day > date.day

    # Text generation according to conditions
    if (condition1 and condition2 and condition3) or (
            condition4 and condition5 and condition6 and condition3 and condition2):
        return f"{vendor} generally charges between on {int(most_frequent_day_dict[vendor])} day of each month " \
               f"invoices are sent. On {date.date()}, an invoice from {vendor} has not been received"
    return None


def no_invoice_received_quaterly(invoice):
    """
    This function performs the no invoice received glean (quarterly case).
    :param pd.DataFrame invoice: invoice data.
    :return: pd.DataFrame of no invoice received glean (quarterly case).
    """
    # Generate quarterly date range (quarterly start) for each vendor
    min_invoice_date = invoice['invoice_date'].min().replace(day=1)
    max_invoice_date = invoice['invoice_date'].max().replace(day=1) + pd.DateOffset(months=1)
    monthly_date_range = pd.date_range(start=min_invoice_date, end=max_invoice_date, freq='QS')
    df = pd.concat((pd.DataFrame({'invoice_quarter_reset': monthly_date_range}).assign(canonical_vendor_id=vendor_id)
                    for vendor_id in invoice['canonical_vendor_id'].unique()),
                   ignore_index=True)

    # Copy to not modify input df
    df2 = invoice.copy()
    # Reset invoice date at the start of the quarter
    df2['invoice_quarter_reset'] = df2['invoice_date'].apply(
        lambda x: x.replace(day=1, month={1: 1, 2: 4, 3: 7, 4: 10}[x.quarter]) if x == x else x)
    df2 = df2[['canonical_vendor_id', 'invoice_date', 'invoice_quarter_reset']]
    # Get the most frequent invoice day (wrt quarter)
    df2['invoice_quarter_day'] = (df2['invoice_date'] - df2['invoice_quarter_reset']).dt.days + 1
    most_frequent_day_dict = df2.groupby(by='canonical_vendor_id')['invoice_quarter_day']. \
        agg(lambda x: x.value_counts().index[0] if x.count() > 0 else 0).to_dict()
    df2.drop_duplicates(subset=['canonical_vendor_id', 'invoice_date'], inplace=True)
    # Keep only the first invoice date of the month
    df2 = df2.groupby(by=['canonical_vendor_id', 'invoice_quarter_reset'])['invoice_date'].agg('min').reset_index()
    # Flag of invoice generated in the corresponding month
    df2['invoice_bool'] = 1
    # Merge with df
    df = df.merge(right=df2,
                  on=['canonical_vendor_id', 'invoice_quarter_reset'],
                  how='left'
                  )
    df['invoice_bool'] = df['invoice_bool'].fillna(value=0)

    # Get the invoice flag for each quarter
    df = df.groupby(by=['canonical_vendor_id', 'invoice_quarter_reset'])['invoice_bool'].agg('max').reset_index()
    # If invoice generated for two consecutive quarters the following variable is 1
    df['consecutive_two_quarter'] = \
        df.groupby(by='canonical_vendor_id')['invoice_bool'].rolling(window=2).mean().reset_index()['invoice_bool']
    # To compare for the next month
    df['consecutive_two_quarter'] = df['consecutive_two_quarter'].shift(1)
    # Get real invoice dates
    df = df.merge(df2[['canonical_vendor_id', 'invoice_quarter_reset', 'invoice_date']],
                  on=['canonical_vendor_id', 'invoice_quarter_reset'],
                  how='left'
                  )

    # Generate daily date range for each vendor
    daily_date_range = pd.date_range(start=min_invoice_date, end=max_invoice_date, freq='D')
    daily_dates = pd.concat((pd.DataFrame({'daily_date': daily_date_range}).assign(canonical_vendor_id=vendor_id)
                             for vendor_id in invoice['canonical_vendor_id'].unique()),
                            ignore_index=True)
    daily_dates['invoice_quarter_reset'] = daily_dates['daily_date'].apply(
        lambda x: x.replace(day=1, month={1: 1, 2: 4, 3: 7, 4: 10}[x.quarter]))
    # Merge with df on each month and vendor
    df = df.merge(right=daily_dates,
                  on=['canonical_vendor_id', 'invoice_quarter_reset'],
                  how='right'
                  )

    # Generate no invoice received glean (quarterly case) text
    df['quarter_daily_day'] = (df['daily_date'] - df['invoice_quarter_reset']).dt.days + 1
    df['invoice_quarter_daily_day'] = (df['invoice_date'] - df['invoice_quarter_reset']).dt.days + 1
    df['most_frequent_invoice_day'] = df['canonical_vendor_id'].map(most_frequent_day_dict)
    df['glean_text'] = df.apply(alarm_no_invoice_quarterly,
                                args=(most_frequent_day_dict,),
                                axis=1)

    # Glean date corresponds to the daily date
    df['glean_date'] = df['daily_date']
    # Glean type --> 4: no_invoice_received
    df['glean_type'] = 4
    # Glean location --> 2: vendor
    df['glean_location'] = 2
    # No invoice id
    df['invoice_id'] = None

    # Only return registers that triggered the glean
    df = df[~df['glean_text'].isna()]
    # Only return variables needed for the output
    columns_to_deliver = ['glean_date', 'glean_text', 'glean_type', 'glean_location', 'invoice_id',
                          'canonical_vendor_id']
    return df[columns_to_deliver]


def alarm_no_invoice_quarterly(row, most_frequent_day_dict):
    """
    Apply this function to each row of the processed data set to obtain the corresponding text.
    :param pd.Series row: row of the no invoice quarterly glean data frame.
    :param dict most_frequent_day_dict: dictionary of most frequent day by vendor.
    :return: str of the corresponding glean text.
    """
    # Variables
    invoice_bool = row['invoice_bool']
    consecutive_two_quarter = row['consecutive_two_quarter']
    invoice_date = row['invoice_date']
    quarter_invoice_day = row['invoice_quarter_daily_day']
    date = row['daily_date']
    d_day = row['quarter_daily_day']
    vendor = row['canonical_vendor_id']

    # Conditions to trigger the no invoice quarterly glean
    condition1 = invoice_bool == 0
    condition2 = consecutive_two_quarter == 1
    condition3 = d_day > most_frequent_day_dict[vendor]
    condition4 = invoice_bool == 1
    condition5 = (invoice_date.quarter == date.quarter) & (invoice_date.year == date.year)
    condition6 = quarter_invoice_day > d_day

    # Text generation according to conditions
    if (condition1 and condition2 and condition3) or (
            condition4 and condition5 and condition6 and condition3 and condition2):
        return f"{vendor} generally charges between on {int(most_frequent_day_dict[vendor])} day of each quarter " \
               f"invoices are sent. On {date.date()}, an invoice from {vendor} has not been received"
    return None


def concat_gleans():
    """
    This function concatenates the generated gleans and generates the output data frame.
    :return: pd.DataFrame of the output.
    """
    # Read raw data sets
    invoice_df, line_item_df = read_raw_data(format_dates=True)

    # Vendor not seen in a while
    glean1 = vendor_not_seen_in_a_while(invoice_df)

    # Accrual alert
    glean2 = accrual_alert(invoice_df, line_item_df)

    # Large month increase mtd
    glean3 = large_month_increase_mtd(invoice_df)

    # No invoice received - monthly
    glean4 = no_invoice_received_monthly(invoice_df)

    # No invoice received - quarterly
    glean5 = no_invoice_received_quaterly(invoice_df)

    # Concatenate gleans
    gleans = [glean1, glean2, glean3, glean4, glean5]
    glean_result = pd.concat(gleans)
    glean_result.reset_index(inplace=True, drop=True)
    # Generate glean_id
    glean_result['glean_id'] = glean_result.index

    return glean_result


if __name__ == "__main__":
    # Concatenate gleans
    result = concat_gleans()

    # Write output csv
    result.to_csv("output.csv", index=False)
