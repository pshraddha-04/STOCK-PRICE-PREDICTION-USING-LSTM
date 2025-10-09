// Load and display prediction results
document.addEventListener('DOMContentLoaded', function() {
    const results = JSON.parse(sessionStorage.getItem('predictionResults'));
    
    if (!results) {
        window.location.href = 'index.html';
        return;
    }
    
    // Update page title and header
    document.querySelector('.header h1').textContent = 'Prediction Results';
    document.querySelector('.header p').textContent = `${results.company_name} (${results.symbol}) Stock Price Prediction for Next ${results.prediction_days} Days`;
    
    // Chart title will be set by Chart.js
    
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
    performanceItems[0].querySelector('.performance-value').textContent = `$${results.rmse}`;
    performanceItems[1].querySelector('.performance-value').textContent = `$${results.mae}`;
    performanceItems[2].querySelector('.performance-value').textContent = results.r2_score;
    
    // Initialize price chart
    initializePriceChart();
    
    // Technical indicators functionality
    let indicatorData = null;
    let charts = {};
    
    document.getElementById('loadIndicators').addEventListener('click', async function() {
        const button = this;
        button.textContent = 'Loading...';
        button.disabled = true;
        
        try {
            const response = await fetch('/indicators', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stockSymbol: results.symbol })
            });
            
            indicatorData = await response.json();
            
            if (indicatorData.success) {
                displaySelectedIndicators();
            } else {
                alert('Error loading indicators: ' + indicatorData.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            button.textContent = 'Load Indicators';
            button.disabled = false;
        }
    });
    
    function displaySelectedIndicators() {
        // Destroy existing charts
        Object.values(charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        charts = {};
        
        // Hide all chart containers
        document.querySelectorAll('.indicator-charts .chart-container').forEach(container => {
            container.style.display = 'none';
        });
        
        // Show selected indicators
        if (document.getElementById('smaCheck').checked) showSMAChart();
        if (document.getElementById('rsiCheck').checked) showRSIChart();
        if (document.getElementById('bbCheck').checked) showBBChart();
        if (document.getElementById('macdCheck').checked) showMACDChart();
    }
    
    function showSMAChart() {
        document.getElementById('smaChart').style.display = 'block';
        const ctx = document.getElementById('smaCanvas').getContext('2d');
        charts.sma = new Chart(ctx, {
            type: 'line',
            data: {
                labels: indicatorData.dates,
                datasets: [{
                    label: 'Close Price',
                    data: indicatorData.close,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2
                }, {
                    label: 'SMA 20',
                    data: indicatorData.sma_20,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: 'SMA 50',
                    data: indicatorData.sma_50,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'Simple Moving Averages' } },
                scales: { y: { beginAtZero: false } }
            }
        });
    }
    
    function showRSIChart() {
        document.getElementById('rsiChart').style.display = 'block';
        const ctx = document.getElementById('rsiCanvas').getContext('2d');
        charts.rsi = new Chart(ctx, {
            type: 'line',
            data: {
                labels: indicatorData.dates,
                datasets: [{
                    label: 'RSI',
                    data: indicatorData.rsi,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'RSI (14)' } },
                scales: { y: { min: 0, max: 100 } }
            }
        });
    }
    
    function showBBChart() {
        document.getElementById('bbChart').style.display = 'block';
        const ctx = document.getElementById('bbCanvas').getContext('2d');
        charts.bb = new Chart(ctx, {
            type: 'line',
            data: {
                labels: indicatorData.dates,
                datasets: [{
                    label: 'Close Price',
                    data: indicatorData.close,
                    borderColor: '#3b82f6',
                    borderWidth: 2
                }, {
                    label: 'Upper Band',
                    data: indicatorData.bb_upper,
                    borderColor: '#ef4444',
                    borderWidth: 1,
                    fill: false
                }, {
                    label: 'Middle Band',
                    data: indicatorData.bb_middle,
                    borderColor: '#f59e0b',
                    borderWidth: 1,
                    fill: false
                }, {
                    label: 'Lower Band',
                    data: indicatorData.bb_lower,
                    borderColor: '#10b981',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'Bollinger Bands' } },
                scales: { y: { beginAtZero: false } }
            }
        });
    }
    
    function showMACDChart() {
        document.getElementById('macdChart').style.display = 'block';
        const ctx = document.getElementById('macdCanvas').getContext('2d');
        charts.macd = new Chart(ctx, {
            type: 'line',
            data: {
                labels: indicatorData.dates,
                datasets: [{
                    label: 'MACD',
                    data: indicatorData.macd,
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: 'Signal',
                    data: indicatorData.macd_signal,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: 'Histogram',
                    data: indicatorData.macd_hist,
                    type: 'bar',
                    backgroundColor: 'rgba(139, 92, 246, 0.3)',
                    borderColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'MACD' } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }
});

function exportToCSV() {
    const results = JSON.parse(sessionStorage.getItem('predictionResults'));
    
    const csvData = [
        ['Metric', 'Value'],
        ['Stock Symbol', results.symbol],
        ['Prediction Days', results.prediction_days],
        ['Current Price', `$${results.current_price}`],
        ['Predicted Price', `$${results.predicted_price}`],
        ['Price Change', `$${results.price_change}`],
        ['Percent Change', `${results.percent_change}%`],
        ['Confidence', `${results.confidence}%`],
        ['RMSE', `$${results.rmse}`],
        ['MAE', `$${results.mae}`],
        ['R² Score', results.r2_score]
    ];
    
    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${results.symbol}_prediction_results.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

function initializePriceChart() {
    const results = JSON.parse(sessionStorage.getItem('predictionResults'));
    
    // Fetch recent price data for chart
    fetch('/indicators', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stockSymbol: results.symbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            // Get last 30 days for display
            const recentDates = data.dates.slice(-30);
            const recentPrices = data.close.slice(-30);
            
            // Add prediction point
            const today = new Date();
            const futureDate = new Date(today);
            futureDate.setDate(today.getDate() + results.prediction_days);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [...recentDates, futureDate.toISOString().split('T')[0]],
                    datasets: [{
                        label: 'Historical Price',
                        data: [...recentPrices, null],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 3,
                        fill: true
                    }, {
                        label: 'Predicted Price',
                        data: [...Array(recentPrices.length).fill(null), results.predicted_price],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        pointRadius: 8,
                        pointBackgroundColor: '#3b82f6'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `${results.company_name} (${results.symbol}) Stock Price Prediction`,
                            font: { size: 18, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'bottom'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Price ($)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        }
                    }
                }
            });
        }
    })
    .catch(error => {
        console.error('Error loading price chart:', error);
    });
}