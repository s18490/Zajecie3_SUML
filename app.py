import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model4.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

#sex_d = {0:"Kobieta",1:"Mężczyzna"}
#pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
#embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem
def main():

	st.set_page_config(page_title="Sprawdź czy wyzdrowiejesz")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://cdn.pixabay.com/photo/2020/10/04/02/15/healthy-5624981_960_720.png")

	with overview:
		st.title("Sprawdź czy wyzdrowiejesz")
	#with left:
	#	sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
	#	pclass_radio = st.radio( "Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
	#	embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

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