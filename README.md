# Nobel Laureate Data Dashboard

An extensive dashboard (or collection of visualizations) with regard to Nobel Laureates and Nobel Prizes, based on Plotly & Dash.
Head to www.nbldata.org for a live dashboard.

- It is a testbed for plot types, so you will find many different plot types, along with the code needed to prepare the data
- It's also about data storytelling; while there is not a single narrative, you will find interestign facts about the prizes.
- The most recent addition is the list generator, which allows you to filter, output and export pretty much any list of Nobel laureates.

- There are also further improvements planned for the future, such as better mobile responsiveness, a complete switch from pandas to polars, a dark mode, unified filters, and much more.

*If you want to use the code:*
The main files are app.py (which runs the Dash/Flask app, and contains the layout and the filter logic), and plotdatagenerator.py (which contains alle the functions for filtering and generating the plots). There are extensive comments in these files.

The structure, filtering logic and Linux server deployment are also covered in a series of Medium articles, which you can find here: https://medium.com/@wolfganghuang


<img width="2002" height="1400" alt="grafik" src="https://github.com/user-attachments/assets/586d31de-db90-4b19-bc02-1a383b632646" />
