import time
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import itertools
from solarradiation_DDR import *
from osgeo import gdal
from osgeo.gdalconst import *
import os 
import pandas as pd
import glob
import warnings

warnings.simplefilter(action='ignore')

# Landsat8 픽셀간 다중선형회귀 진행 
# 주석 : 2022.12.16 작성 --- 이승우 --- 

class Simulation():
    def __init__(self, rad_path, cloud_path, condition_csv, date,d_s, row, col, band, lat1, lat2, dem,pixel_value):

        # csv 통해 모의 영상 데이터 기상 변수 가져오기 
        self.condition_csv = condition_csv  # 기상 데이터 경로
        self.date = date
        self.d_s = d_s

        condition_df = pd.read_csv(self.condition_csv)
        condition_df = condition_df[condition_df["Date(1)"] == self.date]
        self.S_elevation_s = condition_df["Ele. Ang. (7)"].values[0]
        self.temp_s = condition_df["Temp (9)"].values[0]
        self.humid_s = condition_df["Humid(10)"].values[0]
        self.vis_s = condition_df["Vis (11)"].values[0]
        self.rain_s = condition_df["Rain(16)"].values[0]

        self.rad_path = rad_path  # Landsat8 2-7밴드 영상 위치
        self.rad_file = glob.glob(rad_path + "/*.tif")  # 2-7밴드 영상 목록
        self.rad_file = [file for file in glob.glob(rad_path + "/*.tif") if str(date) not in file]
        self.num_f = len(self.rad_file)   # 영상 개수
        self.cloud_path = cloud_path  # 구름 영상 위치 Landsat8 QA_Band file 위치

        self.pixel_value = pixel_value
        self.row = row  # 영상의 행
        self.col = col  # 영상의 열
        self.length = self.row * self.col  # 영상의 길이 행 * 열
        self.band = band  # 영상의 밴드 Landsat 2~7밴드 총 6밴드

        self.lat1 = lat1  # latitude of up conner
        self.lat2 = lat2  # latitude of low conner
        self.de_lat = -(self.lat1 - self.lat2) / (self.row - 1)
        self.lat = np.array([self.lat1 + self.de_lat * float(i) for i in range(self.row)])

        self.dem = gdal.Open(dem).ReadAsArray()[: self.row, : self.col]
        self.data_len = []

        # 추가된 부분
        print(self.date, self.d_s, self.S_elevation_s, self.temp_s, self.humid_s, self.vis_s,self.rain_s, self.num_f)
    def qa_cloud(self,data_arr):
        data_arr = data_arr.reshape(-1)
        cloud_mask_index = []
        for i in range(len(self.pixel_value)):
            tmp_cloud_index = np.where(data_arr == self.pixel_value[i])[0]
            cloud_mask_index.extend(tmp_cloud_index)

        cloud_mask_index = sorted(set(cloud_mask_index))
        return cloud_mask_index 
    
    def image_to_array(self, file_name, coastal = False, cloud=False):
        # Landsat8 영상의 QA밴드 영상과 2~7영상을 받아오는 함수 
        Envi_image = gdal.Open(file_name) # gdal 함수를 이용해 tif 영상 open (rasterio도 사용 가능하다.)
        
        Envi_array = Envi_image.ReadAsArray() 
        
        if len(Envi_array.shape) == 2: # QA밴드 진행
            Envi_array_result = Envi_array[:self.row, :self.col]

        else:
            Envi_array_result = np.zeros((self.row, self.col, self.band), dtype=np.float32)

            if coastal == True:
                print("coastal is True")
                for i in range(self.band):
                    Envi_array_result[:, :, i] = Envi_array[i + 1, :self.row, :self.col]

            elif coastal == False: # 2~7 밴드 영상 진행 코드 
                for i in range(self.band):
                    Envi_array_result[:, :, i] = Envi_array[i, :self.row, :self.col]

        return Envi_array_result
        
    def process_csv(self):
        # 기상 데이터 기온, 습도, 가시거리, 강우량, 직달일사(direct solar radiation), 확산일사(diffuse solar radiation), 반사일사(reflected solar radiation)
        # 직달일사 설명 http://calslab.snu.ac.kr/ncam/board.read?mcode=121111&id=6
        # 직달일사, 확산일사, 반사일사 설명 : https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=atrp00&logNo=220969290475&parentCategoryNo=&categoryNo=13&viewDate=&isShowPopularPosts=true&from=search

        condition = pd.read_csv(self.condition_csv) # 기상 데이터 csv open
        self.atm = [[] for _ in range(4)] # atm -- 기상변수 4개 
        condition = condition[condition["Date(1)"] != self.date]

        self.R = []
        self.Id = []
        self.Ir = []

        date = condition["Date(1)"].values.tolist() # 기상 데이터의 날짜와 영상 날짜의 날짜를 비교하기 위해 기상 데이터의 날짜 변수에 대입
        date = [d for d in date if d != self.date]

        for k in range(0, self.num_f):
            file = self.rad_file[k].split("//")[-1]
            file = file.split("\\")[-1]

            for m in range(len(date)):
                if str(file[:8]) == str(date[m]): # 기상 데이터 날짜와 영상 날짜가 같을 경우만 진행 

                    each_data = condition.iloc[m] # 해당 csv의 날짜의 행(해당 날짜의 기상 데이터)를 불러온다.
                    self.atm[0].append(each_data["Temp (9)"]) # 기온
                    self.atm[1].append(each_data["Humid(10)"]) # 습도
                    self.atm[2].append(each_data["Vis (11)"]) # 가시거리
                    self.atm[3].append(each_data["Rain(16)"]) # 강우량  
                    each_R1, each_Id1, each_Ir1 = solarradiation_DDR(self.dem, self.lat, 30, each_data["day of y(5)"], each_data["Ele. Ang. (7)"])
                    # print(each_R1, each_Id1, each_Ir1)
                    # each, R1, each_Id1, each_Ir1은 각각 직달일사, 확산일사, 반사일사
                    # 받아오기 위해 solarradiation_DDR 함수를 이용한다. 파라미터로는 dem, lat(위도), 30m(landsat), 365일 중의 해당 영상의 날짜(day of y(5)), 태양 고도(Ele. Ang. (7)) 를 넣는다.
                    self.R.append(each_R1.reshape(-1)) # append 하기 위해 전부 1차원으로 변경
                    self.Id.append(each_Id1.reshape(-1))
                    self.Ir.append(each_Ir1.reshape(-1))
        self.R1, self.Id1, self.Ir1 = solarradiation_DDR(self.dem, self.lat, 30, self.d_s, self.S_elevation_s) # 모의 영상을 생성하기 위해 모의 영상의 직달일사, 확산일사, 반사일사 받기
        self.R1 = self.R1.reshape(-1)
        self.Id1 = self.Id1.reshape(-1)
        self.Ir1 = self.Ir1.reshape(-1)
        self.R = np.asarray(self.R) # numpy형태로 변환
        self.Id = np.asarray(self.Id)
        self.Ir = np.asarray(self.Ir)
        print(self.Ir.shape, self.R.shape, self.Id.shape)
        
    def image_simulate(self, save_image_name):

        #################################
        # 영상과 QA밴드 영상을 불러오는 함수

        all_img = []

        for i in range(0, self.num_f):
            file = self.rad_file[i]
            temp_file = self.rad_file[i].split("\\")[-1] # 해당 영상의 날짜 가져오기

            file_cloud = self.cloud_path + "/"+ str(temp_file)[:8] + '_qa_pixel.tif' # qa밴드 경로/해당 영상의 날짜_qa_pixel.tif 
            cloud_mask = Simulation.image_to_array(self, file_cloud, coastal=False, cloud=True) # qa밴드 영상 불러오기

            cloud_mask_index = Simulation.qa_cloud(self,cloud_mask)
            img = Simulation.image_to_array(self, file)
            img = img.reshape(self.length, self.band)

            for b in range(band):
                zero_index = np.where(img[:, b] <= 0)[0] # 기존 영상 밴드의 음수값이 있다면 찾아내기

                img[zero_index, b] = np.nan #  음수값이 존재한다면 nan값으로 대체
                img[cloud_mask_index, b] = np.nan # pixel_clear가 아닌 pixel을 nan값으로 대체 

            all_img.append(img)

        start = time.time()
        self.R = np.reshape(self.R, (len(all_img), self.row, self.col, ))
        self.Id = np.reshape(self.Id, (len(all_img), self.row, self.col))
        self.Ir = np.reshape(self.Ir, (len(all_img), self.row, self.col ))
        self.atm = np.array(self.atm)

        all_x_train = np.zeros((len(all_img), self.row, self.col, 32))

        # 훈련 데이터 전처리 
        for i in range(len(all_img)): # 훈련 데이터만큼 
            tmp_x = all_img[i]
            tmp_x = np.reshape(tmp_x, (self.row, self.col, self.band))

            new_tmp_x = np.zeros((self.row, self.col, 32))

            for r in range(self.row):   
                for c in range(self.col):
                    atm_data = self.atm[:,i:i+1]
                    if r == 0 or r == self.row - 1 or c == 0 or c == self.col - 1: 
                        new_tmp_x[r, c, :4] = atm_data.ravel()[:4]
                        new_tmp_x[r, c, 4] = self.R[i,r, c]
                        new_tmp_x[r, c, 5] = self.Id[i,r, c]
                        new_tmp_x[r, c, 6] = self.Ir[i,r, c]
                        new_tmp_x[r,c, 31] = 1
                    else: # 테두리가 아닐 경우 
                        new_tmp_x[r, c, :4] = atm_data.ravel()[:4]
                        new_tmp_x[r, c, 4:13] = self.R[i,r-1:r+2, c-1:c+2].ravel()
                        new_tmp_x[r, c, 13:22] = self.Id[i,r-1:r+2, c-1:c+2].ravel()
                        new_tmp_x[r, c, 22:31] = self.Ir[i,r-1:r+2, c-1:c+2].ravel()
                        new_tmp_x[r,c, 31] = 1

            all_x_train[i] = new_tmp_x # x_train 54개 영상 학습 
            # 해당 x_train의 차원 확인하고 gkrtmqdp 맞게 pixel마다 54, 8 or 54, 32로 진행해야한다. 
            # y_train은 기존 그대로 
        end = time.time()
        print(f"all_x_train 생성에 걸린 시간: {end - start:.4f}초")
        # x_test a1..a32 설정 
        self.R1 = np.reshape(self.R1, (self.row, self.col))
        self.Id1 = np.reshape(self.Id1, (self.row, self.col))
        self.Ir1 = np.reshape(self.Ir1, (self.row, self.col))

        all_x_test = np.zeros((self.row, self.col, 32))

        for r in range(self.row):
            for c in range(self.col):
                if r == 0 or r == self.row - 1 or c == 0 or c == self.col - 1:
                    atm_data = np.array([self.temp_s, self.humid_s, self.vis_s, self.rain_s])
                    all_x_test[r, c, :4] = atm_data
                    all_x_test[r, c, 4:7] = [self.R1[r, c], self.Id1[r, c], self.Ir1[r, c]]
                    all_x_test[r,c, 31] = 1
                else:
                    atm_data = np.array([self.temp_s, self.humid_s, self.vis_s, self.rain_s])
                    all_x_test[r, c, :4] = atm_data
                    all_x_test[r, c, 4:13] = self.R1[r-1:r+2, c-1:c+2].reshape(-1)
                    all_x_test[r, c, 13:22] = self.Id1[r-1:r+2, c-1:c+2].reshape(-1)
                    all_x_test[r, c, 22:31] = self.Ir1[r-1:r+2, c-1:c+2].reshape(-1)
                    all_x_test[r,c, 31] = 1


        # 모의 영상 만들기 실험 진행 
        new_img = np.zeros((self.row, self.col, self.band), dtype=np.float64) # 모의 영상의 사이즈 생성 (전체 영상 길이, 밴드) numpy float64형태 
        # 학습 과정 y_train 3x3 테두리 과정 결측값 처리 
        error_cnt = 0 
        num_bands = 6

        # Initialize the cnt array with zeros
        band_cnt = np.zeros(num_bands, dtype=int)

        for i in range(self.row):
            print("학습 진행상황 : ", i)
            for j in range(self.col):
                if j == 0 or j == self.col - 1 or i == 0 or i == self.row - 1:
                    y = []
                    for k in range(self.num_f):
                        each_img = all_img[k]
                        each_img = np.reshape(each_img, (self.row, self.col, self.band))
                        y.append(each_img[i,j,:])
                    y = np.array(y)
                    
                    for b in range(self.band):
                        each_y = y[:, b]
                        nan_location = np.isnan(each_y)
                        non_nan_index = np.where(nan_location == False)[0]

                        x_train = all_x_train[:, i, j, [0, 1, 2, 3, 4, 5, 6, 31]]
                        x_train = x_train[non_nan_index]
                        x_train = np.array(x_train)

                        y_train = each_y[non_nan_index]
                        y_train = np.array(y_train)

                        x_test = all_x_test[i, j, [0,1,2,3,4,5,6, 31]]
                        x_test = np.reshape(x_test, (1,-1))

                        try:
                            model = sm.OLS(y_train,x_train).fit()
                            prediction = model.predict(x_test)
                            new_img[i, j, b] = prediction
                        except ValueError as e:
                            error_cnt +=1 
                            print("Error occurred:", str(e))

                else:
                    # 3x3 방법
                    y = []
                    one_y = []
                    for k in range(self.num_f):
                        each_img = all_img[k]
                        each_img = np.reshape(each_img, (self.row, self.col, self.band))

                        y.append(each_img[i-1:i+2, j-1:j+2, :])
                        one_y.append(each_img[i,j,:])
                    y = np.reshape(y, (self.num_f, -1, self.band)) # shape: (54, 9, 6) # 54, 9, 6
                    one_y = np.asarray(one_y)
                    center_y = y[:, 4, :]

                    for b in range(self.band):
                        each_y = center_y[:,b]
                        nan_location = np.isnan(each_y)
                        non_nan_index = np.where(nan_location == False)[0]

                        y_train = each_y[non_nan_index]
                        y_train = np.array(y_train)

                        x_train = all_x_train[:, i, j, :]
                        x_train = x_train[non_nan_index]
                        x_train = np.array(x_train)

                        x_test = all_x_test[i, j, :]
                        x_test = np.reshape(x_test, (1,-1))

                        each_one_y = one_y[:,b]
                        nan_one_location = np.isnan(each_one_y)
                        one_index = np.where(nan_one_location == False)[0]
                        one_y_train = each_one_y[one_index]

                        one_x_train = all_x_train[:, i, j, [0,1,2,3,8,17,26,31]]
                        one_x_train = one_x_train[one_index]

                        one_x_test = all_x_test[i,j, [0,1,2,3,8,17,26,31]]
                        one_x_test = np.reshape(one_x_test, (1,-1))

                        try:
                            model_one = sm.OLS(one_y_train,one_x_train).fit()
                            model_three = sm.OLS(y_train,x_train).fit()

                            one_r2 = model_one.rsquared_adj
                            three_r2 = model_three.rsquared_adj

                            print(x_train.shape, y_train.shape, x_test.shape)
                            print(one_x_train.shape, one_y_train.shape, one_x_test.shape)
                            print(x_train.dtype, y_train.dtype, x_test.dtype, one_x_train.dtype, one_y_train.dtype, one_x_test.dtype)
                            if one_r2 >= three_r2:
                                prediction = model_one.predict(one_x_test)
                                new_img[i,j,b] = prediction
                            else:
                                if b == 0:
                                    band_cnt[0] += 1
                                elif b == 1:
                                    band_cnt[1] += 1
                                elif b == 2:
                                    band_cnt[2] += 1
                                elif b == 3:
                                    band_cnt[3] += 1
                                elif b == 4:
                                    band_cnt[4] += 1
                                elif b == 5:
                                    band_cnt[5] += 1

                                prediction = model_three.predict(x_test)
                                new_img[i,j,b] = prediction

                        except ValueError as e:
                            error_cnt +=1 
                            print("Error occurred:", str(e))
                            
        print("error cnt : ",error_cnt)
        print("band_cnt : ", band_cnt)
        # 완료된 후 npy 파일로 저장
        temp_image = gdal.Open(self.rad_file[0])
        trans = temp_image.GetGeoTransform()
        proj = temp_image.GetProjection()

        outdriver = gdal.GetDriverByName('GTiff')
        outdata = outdriver.Create(save_image_name, self.col, self.row, self.band, gdal.GDT_UInt16)
        outdata.SetGeoTransform(trans)
        outdata.SetProjection(proj)

        for i in range(self.band):
            outdata.GetRasterBand(i + 1).WriteArray(new_img[:, :, i])

        np.save('./r2/20221020_band_cnt.np', band_cnt)
if __name__ == "__main__":

    ##################

    start_time = time.time()
    # Reference image path
    rad_path = "./Landsat_image/115-35_original_subset_band/"
    cloud_path = "./Landsat_image/115-35_original_qa_band/"
    # qa_band pixel_value  
    pixel_value = [0,22080, 22280,24088, 24344, 24216, 24472,30048, 54596, 54852, 55052, 56856, 56984, 57240]
    # read additional information
    condition_csv = './csv/Landsat8_115-35_original.csv'

    # solar information
    date_s = 20221020  # date
    d_s = 293
    # sub image size
    row = 1504
    col = 1497
    band = 6

    # Solar radiation estimation
    lat1 = 36.117779 # latitude of up conner
    lat2 = 35.708568 # latitude of low conner

    # Dem Image
    dem = "./dem_landcover/DEM_115-35_subset.tif"

    # image save path
    save_path = "./Simulation_python/115-35"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_image_name = save_path + "/20221020_r2_dtype64.tif"

    si = Simulation(rad_path, cloud_path, condition_csv, date_s,d_s, row, col, band, lat1, lat2, dem,pixel_value)
    si.process_csv()
    si.image_simulate(save_image_name)