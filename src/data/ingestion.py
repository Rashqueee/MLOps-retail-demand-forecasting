import requests
import os
import time
from datetime import datetime
import pandas as pd

headers = {
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'x-price-center': 'true',
    'sec-ch-ua-mobile': '?0',
    'x-tkpd-akamai': 'pdpMainInfo',
    'accept': '*/*',
    'content-type': 'application/json',
    'x-tkpd-pdpb': '0',
    'Referer': 'https://www.tokopedia.com/',
    'x-source': 'tokopedia-lite',
    'x-device': 'desktop',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
    'x-tkpd-lite-service': 'zeus',
}

# Query GraphQL
graphql_query = """
fragment ProductMedia on pdpDataProductMedia { media { type urlOriginal: URLOriginal urlThumbnail: URLThumbnail urlMaxRes: URLMaxRes videoUrl: videoURLAndroid prefix suffix description variantOptionID __typename } videos { source url __typename } __typename }
fragment ProductHighlight on pdpDataProductContent { name price { value currency priceFmt slashPriceFmt discPercentage __typename } campaign { campaignID campaignType campaignTypeName campaignIdentifier background percentageAmount originalPrice discountedPrice originalStock stock stockSoldPercentage threshold startDate endDate endDateUnix appLinks isAppsOnly isActive hideGimmick showStockBar __typename } thematicCampaign { additionalInfo background campaignName icon __typename } stock { useStock value stockWording __typename } variant { isVariant parentID __typename } wholesale { minQty price { value currency __typename } __typename } isCashback { percentage __typename } isTradeIn isOS isPowerMerchant isWishlist isCOD preorder { duration timeUnit isActive preorderInDays __typename } __typename }
fragment ProductInfo on pdpDataProductInfo { row content { title subtitle applink __typename } __typename }
fragment ProductDetail on pdpDataProductDetail { title productDetailDescription { title content __typename } content { title subtitle applink showAtFront isAnnotation __typename } __typename }
fragment ProductSocial on pdpDataSocialProof { row content { icon title subtitle applink type rating __typename } __typename }
fragment ProductDataInfo on pdpDataInfo { icon title isApplink applink content { icon text __typename } __typename }
fragment ProductCustomInfo on pdpDataCustomInfo { icon title isApplink applink separator description __typename }
fragment ProductVariant on pdpDataProductVariant { errorCode parentID defaultChild sizeChart totalStockFmt variants { productVariantID variantID name identifier option { picture { urlOriginal: url urlThumbnail: url100 __typename } productVariantOptionID variantUnitValueID value hex stock __typename } __typename } children { productID price priceFmt slashPriceFmt discPercentage optionID optionName productName productURL picture { urlOriginal: url urlThumbnail: url100 __typename } stock { stock isBuyable stockWordingHTML minimumOrder maximumOrder __typename } isCOD isWishlist campaignInfo { campaignID campaignType campaignTypeName campaignIdentifier background discountPercentage originalPrice discountPrice stock stockSoldPercentage startDate endDate endDateUnix appLinks isAppsOnly isActive hideGimmick isCheckImei minOrder showStockBar __typename } thematicCampaign { additionalInfo background campaignName icon __typename } ttsPID ttsSKUID __typename } __typename }
fragment ProductCategoryCarousel on pdpDataCategoryCarousel { linkText titleCarousel applink list { categoryID icon title isApplink applink __typename } __typename }
fragment ProductDetailMediaComponent on pdpDataProductDetailMediaComponent { title description contentMedia { url ratio type __typename } show ctaText __typename }
fragment PdpDataComponentShipmentV4 on pdpDataComponentShipmentV4 { data { productID warehouse_info { warehouse_id is_fulfillment district_id postal_code geolocation city_name ttsWarehouseID __typename } useBOVoucher isCOD metadata __typename } __typename }
query PDPMainInfo($productKey: String, $shopDomain: String, $layoutID: String, $extraPayload: String, $queryParam: String, $source: String, $userLocation: pdpUserLocation) { pdpMainInfo(shopDomain: $shopDomain, productKey: $productKey, layoutID: $layoutID, extraPayload: $extraPayload, queryParam: $queryParam, source: $source, userLocation: $userLocation) { requestID extraPayload data { layoutName basicInfo { alias createdAt isQA id: productID shopID shopName minOrder maxOrder weight weightUnit condition status url needPrescription catalogID isLeasing isBlacklisted isTokoNow defaultMediaURL menu { id name url __typename } blacklistMessage { identifier imageURL title description button buttonArea buttonName url supportingImage { url width height __typename } __typename } category { id name title breadcrumbURL isAdult isKyc minAge detail { id name breadcrumbURL isAdult __typename } ttsID ttsDetail { id name breadcrumbURL isAdult __typename } __typename } txStats { transactionSuccess transactionReject countSold paymentVerified itemSoldFmt __typename } stats { countView countReview countTalk rating __typename } productID ttsPID ttsSKUID ttsShopID isAggregatedWithTTS __typename } __typename } components { name type kind position data { ...ProductMedia ...ProductHighlight ...ProductInfo ...ProductDetail ...ProductSocial ...ProductDataInfo ...ProductCustomInfo ...ProductVariant ...ProductCategoryCarousel ...ProductDetailMediaComponent ...PdpDataComponentShipmentV4 __typename } __typename } __typename } }
"""

product_keys = [
    'ekonomi-p0wer-liquid-jruk-nipis-pouch-760-ml-sabun-cuci-piring',
    'blue-band-serbaguna-200g',
    'ultra-uht-cho-1000-ml',
    'bogasari-terigu-segitiga-biru-1-kg',
    'vp-facial-tissue-kiloan-715-s'
]

shop_domain = 'hypermartmalang' 
scraped_data = []
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Memulai proses scraping pada {current_time}...")

for p_key in product_keys:
    json_data = [{
        'operationName': 'PDPMainInfo',
        'variables': {
            'productKey': p_key,
            'shopDomain': shop_domain,
            'layoutID': '',
            'extraPayload': '',
            'queryParam': '',
            'source': 'P1',
            'userLocation': {
                'addressID': '',
                'districtID': '2274',
                'postalCode': '',
                'latlon': '',
                'cityID': '176',
            },
        },
        'query': graphql_query,
    }]

    try:
        response = requests.post('https://gql.tokopedia.com/graphql/PDPMainInfo', headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        
        pdp_main_info = data[0].get('data', {}).get('pdpMainInfo', {})
        if not pdp_main_info:
            print(f"[-] Data tidak ditemukan untuk {p_key}.")
            continue

        basic_info = pdp_main_info.get('data', {}).get('basicInfo', {})
        components = pdp_main_info.get('components', [])

        product_content = {}
        for comp in components:
            if comp.get('name') == 'product_content':
                product_content = comp.get('data', [{}])[0]
                break

        # Ekstraksi Fitur
        product_name = product_content.get('name', p_key) # Mengambil nama asli produk
        price = product_content.get('price', {}).get('value', 0)
        stock = product_content.get('stock', {}).get('value', 0)
        
        tx_stats = basic_info.get('txStats', {})
        sold = tx_stats.get('countSold', 0)
        
        stats = basic_info.get('stats', {})
        rating = stats.get('rating', 0.0)

        scraped_data.append({
            'timestamp': current_time,
            'product_id': basic_info.get('id', ''),
            'product_name': product_name, # Diubah dari product_key menjadi product_name
            'price': int(price) if price else 0,
            'stock': int(stock) if stock else 0,
            'sold': int(sold) if sold else 0,
            'rating': float(rating) if rating else 0.0
        })
        
        print(f"[+] Berhasil scrape: {product_name[:30]}...")

    except Exception as e:
        print(f"[-] Gagal scraping {p_key}: {e}")
        
    # Jeda untuk menghindari anti-bot
    time.sleep(2)

df_demand = pd.DataFrame(scraped_data)

print("\n--- HASIL EKSTRAKSI ---")
print(df_demand.to_string())

# Mengarahkan output ke folder DVC
output_path = 'data/raw.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Simpan dengan mode append
df_demand.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))