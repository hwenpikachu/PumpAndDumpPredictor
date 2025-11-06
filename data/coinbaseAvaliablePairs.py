import pandas as pd
import json
import requests


if __name__ == "__main__":
    """Simple script to query the Coinbase API and determine what cryptocurrency pairs are available.

       Will Save results to a CSV file.
    """

    # https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproducts -->  DOCUMENTATION
    endpoint = 'https://api.exchange.coinbase.com/products'  # this is the coinbase API endpoint for the data
    response = requests.get(endpoint)
    if response.status_code == 200:  # checks if API says our request was OK
        data = json.loads(response.text)  # loads API data response
        # next we create and store the JSON result into a Pandas Dataframe
        data_pd = pd.DataFrame(data)
        data_pd.rename(columns={'id': 'Symbol'}, inplace=True)  # rename one column to be unix
        data_pd.sort_values('Symbol', inplace=True)   # lets sort our results alphabetically
        # Give it a filename and output our results to a CSV
        OUTPUT_FILENAME = "Coinbase_Available_Tradeable_Products.csv"
        data_pd.to_csv(OUTPUT_FILENAME, index=False)
    else:
        print(f"Oops. Bad Response from server. \n {response.status_code}")  # display our error message
                                   