//! Example: Fetch current weather using Open-Meteo (no API key required)

use climate_data::{
    clients::{open_meteo::OpenMeteoClient, ClientConfig, WeatherClient},
    types::Coordinates,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🌍 Climate Data Ingestion Example\n");

    // Configure Open-Meteo client (free, no API key)
    let config = ClientConfig {
        api_key: None,
        base_url: String::new(),
        timeout_secs: 30,
        max_retries: 3,
    };

    let client = OpenMeteoClient::new(config)?;
    println!("✅ Initialized {} client", client.name());

    // Define locations
    let locations = vec![
        ("New York", Coordinates::new(40.7128, -74.0060)),
        ("London", Coordinates::new(51.5074, -0.1278)),
        ("Tokyo", Coordinates::new(35.6762, 139.6503)),
        ("Sydney", Coordinates::new(-33.8688, 151.2093)),
    ];

    // Fetch current weather for each location
    println!("\n📊 Current Weather:\n");

    for (name, location) in locations {
        match client.fetch_current(location).await {
            Ok(weather) => {
                println!("🌡️  {}", name);
                println!("   Temperature: {:.1}°C", weather.temperature);
                println!("   Humidity: {:.0}%", weather.humidity);
                println!("   Pressure: {:.1} hPa", weather.pressure);
                println!("   Wind: {:.1} m/s at {:.0}°", weather.wind_speed, weather.wind_direction);
                println!("   Cloud Cover: {:.0}%", weather.cloud_cover);
                if let Some(uv) = weather.uv_index {
                    println!("   UV Index: {:.1}", uv);
                }
                println!();
            }
            Err(e) => {
                eprintln!("❌ Error fetching weather for {}: {}", name, e);
            }
        }
    }

    // Fetch 24-hour forecast for New York
    println!("📈 24-Hour Forecast for New York:\n");

    let ny_location = Coordinates::new(40.7128, -74.0060);
    match client.fetch_forecast(ny_location, 24).await {
        Ok(forecast) => {
            println!("   Total forecast points: {}", forecast.forecasts.len());
            println!("   Generated at: {}", forecast.generated_at);
            println!("   Data source: {}", forecast.source);

            // Show first 6 hours
            println!("\n   First 6 hours:");
            for (i, weather) in forecast.forecasts.iter().take(6).enumerate() {
                println!(
                    "   {}h: {:.1}°C, {:.0}% humidity, {:.1} m/s wind",
                    i,
                    weather.temperature,
                    weather.humidity,
                    weather.wind_speed
                );
            }
        }
        Err(e) => {
            eprintln!("❌ Error fetching forecast: {}", e);
        }
    }

    println!("\n✨ Example complete!");

    Ok(())
}
