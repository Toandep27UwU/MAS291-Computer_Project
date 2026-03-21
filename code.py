import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 0. ĐỌC DỮ LIỆU TỪ FILE CSV (Đã chỉnh lấy đúng dòng 302 đến 314)
# ==========================================
# Đọc toàn bộ file ZHVI.csv
df = pd.read_csv('ZHVI.csv', header=None)

# Cắt lấy đúng phần dữ liệu từ dòng 302 đến 314 trong Excel
# Python đếm từ 0 nên lấy index từ 301 đến 314 (không bao gồm 314)
df_filtered = df.iloc[301:314, :]

# Lấy dữ liệu: Cột B (Florida) là index 1, Cột C (California) là index 2
fl_prices = pd.to_numeric(df_filtered.iloc[:, 1]).values
ca_prices = pd.to_numeric(df_filtered.iloc[:, 2]).values

# Tính toán số tháng tăng giá (cho Slide 2 và 4)
diffs_ca = np.diff(ca_prices)
diffs_fl = np.diff(fl_prices)

x_ca = np.sum(diffs_ca > 0) # Số tháng CA tăng giá
x_fl = np.sum(diffs_fl > 0) # Số tháng FL tăng giá
n_thang_so_sanh = len(diffs_ca) # Sẽ là 12 (vì có 13 tháng)

print("ĐANG XỬ LÝ DỮ LIỆU TỪ THÁNG 1/2025 ĐẾN 1/2026...\n")

# ==========================================
# SLIDE 1: KIỂM ĐỊNH 1 TRUNG BÌNH (California)
# ==========================================
# Giả thuyết: H0: mu = 750000 (Giả định trung bình giá nhà CA là 750k)
mu0_s1 = 750000
n1_ca = len(ca_prices)
mean_ca = np.mean(ca_prices)
std_ca = np.std(ca_prices, ddof=1)

t_stat_1 = (mean_ca - mu0_s1) / (std_ca / math.sqrt(n1_ca))
p_value_1 = 2 * (1 - stats.t.cdf(abs(t_stat_1), df=n1_ca-1))

t_crit_1 = stats.t.ppf(1 - 0.05/2, df=n1_ca-1)
margin_1 = t_crit_1 * (std_ca / math.sqrt(n1_ca))
ci_lower_1, ci_upper_1 = mean_ca - margin_1, mean_ca + margin_1

conclusion_1 = "Reject H0" if p_value_1 < 0.05 else "Fail to reject H0"

print("=== 1. Conduct a hypothesis test and construct a confidence interval for the population mean. (California) ===")
print(f"KTC 95% cho mean: [{ci_lower_1:,.0f}, {ci_upper_1:,.0f}] | mean={mean_ca:,.0f}, s={std_ca:,.0f}, n={n1_ca}, t_alpha/2={t_crit_1:.4f}")
print(f"Kiểm định H0: mu = {mu0_s1}, t0 = {t_stat_1:.4f} \t\t\t Kết luận: {conclusion_1}\n")

# ==========================================
# SLIDE 2: KIỂM ĐỊNH 1 TỶ LỆ (California)
# ==========================================
# Giả thuyết: H0: p = 0.50 (Giả định tỷ lệ tháng tăng giá là 50%)
p0_s2 = 0.50
p_hat_ca = x_ca / n_thang_so_sanh

z_crit_2 = stats.norm.ppf(1 - 0.05/2)
se_ci_2 = math.sqrt((p_hat_ca * (1 - p_hat_ca)) / n_thang_so_sanh)
ci_lower_2, ci_upper_2 = p_hat_ca - z_crit_2*se_ci_2, p_hat_ca + z_crit_2*se_ci_2

se_test_2 = math.sqrt((p0_s2 * (1 - p0_s2)) / n_thang_so_sanh)
z_stat_2 = (p_hat_ca - p0_s2) / se_test_2
p_value_2 = 2 * (1 - stats.norm.cdf(abs(z_stat_2)))

conclusion_2 = "Reject H0" if p_value_2 < 0.05 else "Fail to reject H0"

print("=== 2. Conduct a hypothesis test and construct a confidence interval for the population proportion. (California) ===")
print(f"Số tháng tăng/so sánh = {x_ca}/{n_thang_so_sanh} \t (p_hat = {p_hat_ca:.3f})")
print(f"KTC 95% cho p: [{ci_lower_2:.3f}, {ci_upper_2:.3f}]")
print(f"Kiểm định H0: p = {p0_s2:.2f} \t\t\t\t\t Kết luận: {conclusion_2}\n")

# ==========================================
# SLIDE 3: KIỂM ĐỊNH KHÁC BIỆT 2 TRUNG BÌNH (CA vs FL)
# ==========================================
mean_fl = np.mean(fl_prices)
std_fl = np.std(fl_prices, ddof=1)
n1_fl = len(fl_prices)

t_stat_3, p_value_3 = stats.ttest_ind(ca_prices, fl_prices, equal_var=False)

# KTC Welch
var_ca, var_fl = std_ca**2, std_fl**2
df_3 = (var_ca/n1_ca + var_fl/n1_fl)**2 / ((var_ca/n1_ca)**2/(n1_ca-1) + (var_fl/n1_fl)**2/(n1_fl-1))
t_crit_3 = stats.t.ppf(1 - 0.05/2, df_3)
margin_3 = t_crit_3 * math.sqrt(var_ca/n1_ca + var_fl/n1_fl)
diff_mean = mean_ca - mean_fl

conclusion_3 = "Reject H0" if p_value_3 < 0.05 else "Fail to reject H0"

print("=== 3. Conduct a hypothesis test and construct a confidence interval for the difference between two population means. (CA vs FL) ===")
print(f"diff(mean) = {diff_mean:,.0f} | KTC 95%: [{diff_mean - margin_3:,.0f}, {diff_mean + margin_3:,.0f}]")
print(f"Kiểm định H0: mu_CA - mu_FL = 0, t0 = {t_stat_3:.3f} \t\t Kết luận: {conclusion_3}\n")

# ==========================================
# SLIDE 4: KIỂM ĐỊNH KHÁC BIỆT 2 TỶ LỆ (CA vs FL)
# ==========================================
p_hat_fl = x_fl / n_thang_so_sanh
diff_p = p_hat_ca - p_hat_fl

se_ci_4 = math.sqrt((p_hat_ca*(1-p_hat_ca)/n_thang_so_sanh) + (p_hat_fl*(1-p_hat_fl)/n_thang_so_sanh))
z_crit_4 = stats.norm.ppf(1 - 0.05/2)
ci_lower_4, ci_upper_4 = diff_p - z_crit_4*se_ci_4, diff_p + z_crit_4*se_ci_4

p_pool = (x_ca + x_fl) / (n_thang_so_sanh + n_thang_so_sanh)
se_test_4 = math.sqrt(p_pool * (1 - p_pool) * (2/n_thang_so_sanh))

if se_test_4 == 0:
    z_stat_4, p_value_4 = 0, 1
else:
    z_stat_4 = diff_p / se_test_4
    p_value_4 = 2 * (1 - stats.norm.cdf(abs(z_stat_4)))

conclusion_4 = "Reject H0" if p_value_4 < 0.05 else "Fail to reject H0"

print("=== 4. Conduct a hypothesis test and construct a confidence interval for the difference between two population proportions. ===")
print(f"p1 (CA) = {p_hat_ca:.3f}, p2 (FL) = {p_hat_fl:.3f}, diff = {diff_p:.3f}")
print(f"z={z_stat_4:.3f}, p={p_value_4:.4f} | KTC 95%: [{ci_lower_4:.3f}, {ci_upper_4:.3f}] \t Kết luận: {conclusion_4}\n")

# ==========================================
# SLIDE 5: PHÂN TÍCH HỒI QUY VÀ VẼ BIỂU ĐỒ
# ==========================================
thang_truc_x = np.arange(1, len(ca_prices) + 1)

res_ca = stats.linregress(thang_truc_x, ca_prices)
res_fl = stats.linregress(thang_truc_x, fl_prices)

print("=== 5. Regression analysis ===")
print(f"[California] y = {res_ca.intercept:.2f} + {res_ca.slope:.2f} * x;  R^2 = {res_ca.rvalue**2:.3f}, p = {res_ca.pvalue:.4f}, n = {len(thang_truc_x)}")
print(f"[Florida]    y = {res_fl.intercept:.2f} + {res_fl.slope:.2f} * x;  R^2 = {res_fl.rvalue**2:.3f}, p = {res_fl.pvalue:.4f}, n = {len(thang_truc_x)}")

# Vẽ hình
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Định nghĩa sẵn nhãn và vị trí để dùng chung cho cả 2 biểu đồ
nhan_thang_nam = ['1/2025', '2/2025', '3/2025', '4/2025', '5/2025', '6/2025', 
                  '7/2025', '8/2025', '9/2025', '10/2025', '11/2025', '12/2025', '1/2026']
vi_tri_cu = range(1, 14) 

# --- BIỂU ĐỒ 1: CALIFORNIA (ax1) ---
ax1.plot(thang_truc_x, ca_prices, 'o-', label="Giá trị thực")
ax1.plot(thang_truc_x, res_ca.intercept + res_ca.slope*thang_truc_x, '-', label="Đường xu hướng")
ax1.set_title(f"Xu hướng tuyến tính - California (R²={res_ca.rvalue**2:.3f}, p={res_ca.pvalue:.3f})")
ax1.set_xlabel("Thời gian") # Đổi tên trục x cho hợp với nhãn mới
ax1.set_ylabel("Giá trị ($)")
ax1.legend()
# Chỉnh nhãn trục x cho riêng California
ax1.set_xticks(vi_tri_cu)
ax1.set_xticklabels(nhan_thang_nam, rotation=45)

# --- BIỂU ĐỒ 2: FLORIDA (ax2) ---
ax2.plot(thang_truc_x, fl_prices, 'o-', label="Giá trị thực")
ax2.plot(thang_truc_x, res_fl.intercept + res_fl.slope*thang_truc_x, '-', label="Đường xu hướng")
ax2.set_title(f"Xu hướng tuyến tính - Florida (R²={res_fl.rvalue**2:.3f}, p={res_fl.pvalue:.3f})")
ax2.set_xlabel("Thời gian") # Đổi tên trục x cho hợp với nhãn mới
ax2.set_ylabel("Giá trị")
ax2.legend()
# Chỉnh nhãn trục x cho riêng Florida
ax2.set_xticks(vi_tri_cu)
ax2.set_xticklabels(nhan_thang_nam, rotation=45)

# Đảm bảo layout không bị lẹm viền khi nghiêng chữ 45 độ
plt.tight_layout()

# Lưu thành ảnh
plt.savefig('bieu_do_mas291.png', bbox_inches='tight')