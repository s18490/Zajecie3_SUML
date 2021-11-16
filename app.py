import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()

filename = "model4.sv"
model = pickle.load(open(filename,'rb'))
def main():

	st.set_page_config(page_title="Sprawdź czy wyzdrowiejesz")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://cdn.pixabay.com/photo/2020/10/04/02/15/healthy-5624981_960_720.png")

	with overview:
		st.title("Sprawdź czy wyzdrowiejesz")

	with left:
		objawy_slider = st.slider("Objawy", value=1, min_value=1, max_value=5)
		wiek_slider = st.slider( "Wiek", value=18, min_value=0, max_value=100)
		choroby_slider = st.slider( "Choroby", min_value=0, max_value=5)
		wzrost_slider = st.slider( "Wzrost", min_value=0, max_value=200)

	data = [[objawy_slider, wiek_slider,  choroby_slider, wzrost_slider]]
	zdrowie = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.header("Czy wyzdrowiejesz? {0}".format("Nie" if zdrowie[0] == 1 else "Tak"))
		st.subheader("Pewność predykcji {0:.1f} %".format(s_confidence[0][zdrowie][0] * 100))

if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
