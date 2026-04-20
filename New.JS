async function getWeather() {
    const location = document.getElementById('locationInput').value;
    try {
        const response = await fetch(`/api/weather?location=${location}`);
        const data = await response.json();
        displayWeather(location, data);
    } catch (error) {
        console.error('Error fetching weather:', error);
    }
}