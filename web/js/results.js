// Load and display prediction results
document.addEventListener('DOMContentLoaded', function() {
    const results = JSON.parse(sessionStorage.getItem('predictionResults'));
    
    if (!results) {
        window.location.href = 'index.html';
        return;
    }
    
    // Update page title and header
    document.querySelector('.header h1').textContent = 'Prediction Results';
    document.querySelector('.header p').textContent = `${results.symbol} Stock Price Prediction for Next ${results.prediction_days} Days`;
    
    // Update chart title
    document.querySelector('.prediction-chart text').textContent = `${results.symbol} Stock Price Prediction`;
    
    // Update metric cards
    const currentPriceCard = document.querySelector('.metric-card:nth-child(1)');
    currentPriceCard.querySelector('.metric-value').textContent = `$${results.current_price}`;
    
    const predictedPriceCard = document.querySelector('.metric-card:nth-child(2)');
    predictedPriceCard.querySelector('.metric-value').textContent = `$${results.predicted_price}`;
    predictedPriceCard.querySelector('.metric-change').textContent = `${results.percent_change > 0 ? '+' : ''}${results.percent_change}%`;
    predictedPriceCard.querySelector('.metric-change').className = `metric-change ${results.percent_change >= 0 ? 'positive' : 'negative'}`;
    
    const confidenceCard = document.querySelector('.metric-card:nth-child(3)');
    confidenceCard.querySelector('.metric-value').textContent = `${results.confidence}%`;
    
    // Update performance metrics
    const performanceItems = document.querySelectorAll('.performance-item');
    performanceItems[0].querySelector('.performance-value').textContent = results.rmse;
    performanceItems[1].querySelector('.performance-value').textContent = results.mae;
    performanceItems[2].querySelector('.performance-value').textContent = results.r2_score;
});