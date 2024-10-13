import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import streamlit as st
from streamlit_folium import st_folium

st.title("Kävely")
st.write("Selaathan sivun loppuun asti. Numeeriset arvot löytyvät sivun pohjalta. Jostain syystä kartan alle generoituu 1000 pikseliä tyhjää, jonka jälkeen löydät seuraavat tiedot:")
st.write("Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta:  1962.0")
st.write("Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella: 1958.0")
st.write("Keskinopeus: 5.19 km/h")
st.write("Kuljettu matka: 1.38 km")
st.write("Askeleen keskipituus on ollut suodatetun datan perusteella 0.7 m")
st.write("Askeleen keskipituus on ollut fourier analyysin perusteella 0.71 m")


def suodatetutAskeleet():
    df_step = pd.read_csv('LinearAcceleration.csv')

    # Suodatetaan datasta selvästi kävelytaajuutta suurempitaajuuksiset vaihtelut pois
    # Filtteri: 
    from scipy.signal import butter, filtfilt
    def butter_lowpass_filter(data, cutoff, nyq, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Filttereiden parametrit:
    T = df_step['Time (s)'][len(df_step['Time (s)'])-1] - df_step['Time (s)'][0] # Koko datan pituus
    n = len(df_step['Time (s)']) # Datapisteiden lukumäärä
    fs = n/T # Näytteenottotaajuus (olettaen vakioksi)
    nyq = fs/2 # Nyqvistin taajuus
    order = 3 # Kertaluku
    cutoff = 1/(0.25) # Cut-off taajuus 

    filtered_signal = butter_lowpass_filter(df_step['Linear Acceleration z (m/s^2)'], cutoff, nyq, order)

    df_step['filtered_signal'] = np.zeros(len(df_step)) # Alustetaan sarake etäisyydelle

    for i in range(0, (len(df_step) -1)):
        df_step.loc[i, 'filtered_signal'] = filtered_signal[i] 

    # Suodatetun kiihdytysdatan kuvaaja
    st.write("Suodatetun kiihdytysdatan kuvaaja")
    st.line_chart(df_step, x = 'Time (s)', y = 'filtered_signal', x_label = 'Time', y_label = 'Suodatettu kiihtyvyys')

    # Lasketaan jaksojen määrä signaalissa (ja sitä kautta askelten määrä) laskemalla signaalin nollakohtien ylitysten määrä
    # Nollakohta ylitetään kaksi kertaa jokaisen jakson aikana (ylös- ja alaspäin mentäessä)
    jaksot = 0
    for i in range(len(filtered_signal)-1):
        if(filtered_signal[i] / filtered_signal[i+1] < 0):
            jaksot = jaksot + 1

    return [np.floor(jaksot/2)]

def fourierAnalyysiAskeleet():
    df = pd.read_csv('LinearAcceleration.csv')
    f = df['Linear Acceleration z (m/s^2)']     # Valittu signaali
    t = df['Time (s)']                          # Aika
    N = len(df)                                 # Havaintojen määrä
    dt = np.max(t/len(t))                       # Oletetaan sämpäystaajuus vakioksi

    fourier = np.fft.fft(f,N)                   # Fourier muunnos
    psd = fourier * np.conj(fourier)/N          # Tehospektri
    freq = np.fft.fftfreq(N,dt)                 # Taajuudet
    L = np.arange(1,int(N/2))                   # Rajataan pois nollataajuus ja negatiiviset taajuudet

    df = pd.DataFrame(
        {
            "Freq": freq[L],
            "Psd": psd[L].real
        }
    )

    # Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys
    st.write("Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys")
    st.line_chart(df, x = 'Freq', y = 'Psd', x_label = 'Taajuus', y_label = 'Teho')

    return [np.round(freq[L][psd[L]==np.max(psd[L])][0]*np.max(t), 2), freq[L], psd[L].real]

def haversine(lon1, lat1, lon2, lat2): # Kahden pisteen koordinaatit
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Rarios of Earth in kilometers
    return c * r # Palauttaa koordinaattien välisen etäisyyden maapallon pintaa pitkin

def GPS():
    st.write("Kävelyn reitti kartalla")
    df = pd.read_csv("Location.csv")

    start_lat = df['Latitude (°)'].mean()
    start_long = df['Longitude (°)'].mean()

    map = folium.Map(location = [start_lat, start_long], zoom_start = 14)

    folium.PolyLine(df[['Latitude (°)', 'Longitude (°)']], color='blue', weight = 3.5, opacity = 1).add_to(map)
    st_map = st_folium(map, width=900, height=650)

    # Määritellään latitudi ja longitudi erikseen
    lat = df['Latitude (°)']
    lon = df['Longitude (°)']

    df['dist'] = np.zeros(len(df)) # Alustetaan sarake etäisyydelle

    for i in range(0, (len(df) -1)):
        df.loc[i, 'dist'] = haversine(lon[i], lat[i],  lon[i+1],lat[i+1]) # Peräkkäisten pisteiden välimatka

    df['tot_dist'] = np.cumsum(df['dist']) # Kokonaismatka numpyn cumulative sum funktiolla
    #print(df['tot_dist'].tail(1)) # Tulostaa data framen viimeisen rivin

    return ([ df['tot_dist'].tail(1), df['Time (s)'].tail(1), df['Latitude (°)'], df['Longitude (°)'] ])

askeleet_s = suodatetutAskeleet()
askeleet_f = fourierAnalyysiAskeleet()

askelMaara_s = askeleet_s[0]
askelMaara_f = askeleet_f[0]

GPS_data = GPS()
matka = GPS_data[0][GPS_data[0].last_valid_index()]
aika = GPS_data[1][GPS_data[1].last_valid_index()]

aika = aika/3600 # Muutetaan aika tunneiksi
keskinopeus = matka/aika

askeleenPituus_s = np.round(matka*1000/askelMaara_s, 2)
askeleenPituus_f = np.round( matka*1000/askelMaara_f, 2)

# Tekstimuotoiset raportit
st.write("Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta: ", askelMaara_s)
st.write("Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella: ", askelMaara_f)
st.write("Keskinopeus: ", np.round(keskinopeus, 2), 'km/h')
st.write("Kuljettu matka: ", np.round(matka, 2), "km")
st.write('Askeleen keskipituus on ollut suodatetun datan perusteella', askeleenPituus_s, 'm')
st.write('Askeleen keskipituus on ollut fourier analyysin perusteella ', askeleenPituus_f, 'm' )