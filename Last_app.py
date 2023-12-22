
import numpy as np
import pandas as pd
import joblib
import streamlit as st

model= joblib.load('Final_Model.pkl')

inputs=joblib.load('Final_Input.pkl')

st.title('Restutant Success Prediction\n')
def prediction(online_order, book_table, votes, location, rest_type, cuisines, cost, types, city):
    df=pd.DataFrame(columns=inputs)
    df.at[0,'online_order']= online_order
    df.at[0,'book_table']= book_table
    df.at[0,'votes']= votes
    df.at[0,'location']= location
    df.at[0,'rest_type']= rest_type
    df.at[0,'cuisines']= cuisines
    df.at[0,'cost']= cost
    df.at[0,'types']= types
    df.at[0,'city']= city
    result=model.predict(df)
    return result
    
def main():
    st.title("Success prediction")
    online_order=st.selectbox('online_order', ['Yes', 'No'])
    book_table=st.selectbox('book_table', ['Yes', 'No'])
    votes=st.slider('votes', min_value=0 , max_value=17000 , step=1 , value=0)
    location=st.selectbox('location', ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
                                       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
                                       'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
                                       'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli',
                                       'CV Raman Nagar', 'Electronic City', 'Wilson Garden',
                                       'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
                                       'Marathahalli', 'Koramangala 7th Block', 'Bellandur',
                                       'Sarjapur Road', 'Whitefield', 'East Bangalore',
                                       'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block',
                                       'Frazer Town', 'MG Road', 'Brigade Road', 'Lavelle Road',
                                       'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar',
                                       'Infantry Road', 'St. Marks Road', 'Cunningham Road',
                                       'Race Course Road', 'Commercial Street', 'Vasanth Nagar', 'Domlur',
                                       'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
                                       'Old Madras Road', 'Seshadripuram', 'Kammanahalli',
                                       'Koramangala 6th Block', 'Majestic', 'Langford Town',
                                       'Central Bangalore', 'Sanjay Nagar', 'Brookefield',
                                       'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',
                                       'Koramangala 2nd Block', 'Koramangala 3rd Block',
                                       'Koramangala 4th Block', 'Koramangala', 'Hosur Road',
                                       'Rajajinagar', 'RT Nagar', 'Banaswadi', 'North Bangalore',
                                       'Nagawara', 'Hennur', 'Kalyan Nagar', 'HBR Layout',
                                       'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal',
                                       'Kengeri', 'New BEL Road', 'Sankey Road', 'Malleshwaram',
                                       'Sadashiv Nagar', 'Basaveshwara Nagar', 'Yeshwantpur',
                                       'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar',
                                       'Jalahalli', 'Nagarbhavi', 'Peenya', 'KR Puram'])
    rest_type=st.selectbox('rest_type',['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
                                        'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe',
                                        'Cafe, Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
                                        'Bakery, Dessert Parlor', 'Pub', 'Bakery', 'Takeaway, Delivery',
                                        'Fine Dining', 'Beverage Shop', 'Sweet Shop', 'Bar',
                                        'Dessert Parlor, Sweet Shop', 'Bakery, Quick Bites',
                                        'Sweet Shop, Quick Bites', 'Kiosk', 'Food Truck',
                                        'Quick Bites, Dessert Parlor', 'Beverage Shop, Quick Bites',
                                        'Beverage Shop, Dessert Parlor', 'Takeaway', 'Pub, Casual Dining',
                                        'Casual Dining, Bar', 'Dessert Parlor, Beverage Shop',
                                        'Quick Bites, Bakery', 'Microbrewery, Casual Dining', 'Lounge',
                                        'Bar, Casual Dining', 'Food Court', 'Cafe, Bakery', 'Dhaba',
                                        'Quick Bites, Sweet Shop', 'Microbrewery',
                                        'Food Court, Quick Bites', 'Quick Bites, Beverage Shop',
                                        'Pub, Bar', 'Casual Dining, Pub', 'Lounge, Bar',
                                        'Dessert Parlor, Quick Bites', 'Food Court, Dessert Parlor',
                                        'Casual Dining, Sweet Shop', 'Food Court, Casual Dining',
                                        'Casual Dining, Microbrewery', 'Lounge, Casual Dining',
                                        'Cafe, Food Court', 'Beverage Shop, Cafe', 'Cafe, Dessert Parlor',
                                        'Dessert Parlor, Cafe', 'Dessert Parlor, Bakery',
                                        'Microbrewery, Pub', 'Bakery, Food Court', 'Club',
                                        'Quick Bites, Food Court', 'Bakery, Cafe', 'Pub, Cafe',
                                        'Casual Dining, Irani Cafee', 'Fine Dining, Lounge',
                                        'Bar, Quick Bites', 'Confectionery', 'Pub, Microbrewery',
                                        'Microbrewery, Lounge', 'Fine Dining, Microbrewery',
                                        'Fine Dining, Bar', 'Dessert Parlor, Kiosk', 'Bhojanalya',
                                        'Casual Dining, Quick Bites', 'Cafe, Bar', 'Casual Dining, Lounge',
                                        'Bakery, Beverage Shop', 'Microbrewery, Bar', 'Cafe, Lounge',
                                        'Bar, Pub', 'Lounge, Cafe', 'Club, Casual Dining',
                                        'Quick Bites, Mess', 'Quick Bites, Meat Shop',
                                        'Quick Bites, Kiosk', 'Lounge, Microbrewery',
                                        'Food Court, Beverage Shop', 'Dessert Parlor, Food Court',
                                        'Bar, Lounge'])
    cuisines=st.selectbox('cuisines',['North Indian, Mughlai, Chinese','Chinese, North Indian, Thai',
                                      'Cafe, Mexican, Italian','South Indian, North Indian',
                                      'North Indian, Rajasthani','North Indian',
                                      'North Indian, South Indian, Andhra, Chinese','Pizza, Cafe, Italian',
                                      'Cafe, Italian, Continental','Cafe, Mexican, Italian, Momos, Beverages',
                                      'Cafe','Cafe, Chinese, Continental, Italian','Cafe, Continental',
                                      'Cafe, Fast Food, Continental, Chinese, Momos','Chinese, Cafe, Italian',
                                      'Cafe, Italian, American','Cafe, French, North Indian',
                                      'Cafe, Pizza, Fast Food, Beverages',
                                      'Cafe, Fast Food','Italian, Fast Food, Cafe, European',
                                      'Cafe, Bakery','Cafe, South Indian','Cafe, Fast Food, Beverages',
                                      'North Indian, Cafe, Chinese, Fast Food','Cafe, Italian',
                                      'North Indian, Fast Food, Chinese, Burger','Bakery, Desserts','Pizza',
                                      'North Indian, Biryani, Fast Food',
                                      'Biryani','North Indian, Chinese, Fast Food','Chinese, Thai, Momos',
                                      'North Indian, Mughlai, South Indian, Chinese','South Indian',
                                      'Street Food, Fast Food','Burger, Fast Food','Pizza, Fast Food',
                                      'North Indian, Continental, Italian','North Indian, Chinese',
                                      'North Indian, Chinese, Biryani, Rolls','Chinese, Thai',
                                      'North Indian, Chinese, Momos, Rolls','Fast Food, Street Food, Beverages',
                                      'Ice Cream, Desserts','Biryani, North Indian, Chinese, Andhra, South Indian',
                                      'Italian, Continental, Fast Food, Chinese, Momos',
                                      'Healthy Food, Chinese, Biryani, North Indian, Continental, Salad, American, Burger',
                                      'Biryani, Fast Food',
                                      'Asian, Korean, Indonesian, Japanese, Chinese, Thai, Momos',
                                      'Fast Food, Burger','Desserts, Beverages','Italian, North Indian, Mexican',
                                      'Goan, Seafood, North Indian, Chinese, Biryani','Chinese',
                                      'Bakery','North Indian, Kebab, Chinese, Fast Food',
                                      'Continental, Italian, Mexican, North Indian, Chinese, Steak',
                                      'Burger, Fast Food, Beverages','Italian, North Indian, Biryani',
                                      'Biryani, Chinese, Kebab','Chinese, Continental, Italian, North Indian',
                                      'Fast Food, Rolls, Momos','Biryani, South Indian',
                                      'Fast Food','South Indian, Chinese, North Indian',
                                      'Seafood, North Indian, Chinese, Andhra, Biryani, Kebab',
                                      'Beverages, Desserts, Ice Cream','Mithai, Street Food',
                                      'Biryani, North Indian, Chinese','Desserts','Ice Cream',
                                      'South Indian, North Indian, Chinese','Fast Food, Bakery',
                                      'North Indian, Iranian','Fast Food, Street Food',
                                      'Ice Cream, Desserts, Beverages, Sandwich','South Indian, Biryani',
                                      'Biryani, North Indian, Chinese, Fast Food','Sandwich, Pizza, Beverages',
                                      'Chinese, North Indian','South Indian, North Indian, Chinese, Street Food',
                                      'Mangalorean, South Indian, North Indian','Fast Food, South Indian',
                                      'Biryani, Rolls, Chinese','South Indian, Beverages',
                                      'South Indian, North Indian, Chinese, Beverages',
                                      'South Indian, Chinese, North Indian, Juices',
                                      'North Indian, Kebab, Biryani, Chinese','Chinese, North Indian, Mughlai, Rolls',
                                      'North Indian, Mughlai, Fast Food',
                                      'Desserts, Ice Cream, Beverages, Fast Food, Sandwich',
                                      'North Indian, Chinese, Biryani, Beverages','Beverages, Juices, Ice Cream',
                                      'Chinese, Seafood, Biryani',
                                      'Desserts, Mithai','Cafe, Beverages',
                                      'South Indian, Chinese','Thai, Vietnamese, Asian, Chinese',
                                      'Beverages, Ice Cream','South Indian, North Indian, Chinese, Juices',
                                      'Italian, Pizza',
                                      'North Indian, Fast Food, Rolls','Salad, Healthy Food, Sandwich, Juices, Burger, Desserts, Pizza',
                                      'North Indian, Chinese, Biryani, Kebab'])
    cost=st.slider('cost', min_value=1 , max_value=950 , step=1 , value=1)
    types=st.selectbox('types', ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
                                 'Drinks & nightlife', 'Pubs and bars'])
    city=st.selectbox('city', ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
                               'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
                               'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
                               'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
                               'Koramangala 4th Block', 'Koramangala 5th Block',
                               'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
                               'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
                               'Old Airport Road', 'Rajajinagar', 'Residency Road',
                               'Sarjapur Road', 'Whitefield'])
    if st.button('Predict'):
        result = prediction(online_order, book_table, votes, location, rest_type, cuisines, cost, types, city)
        st.text(result)
        st.caption('1=Highly rated, 0=Low rated')
        
main()
