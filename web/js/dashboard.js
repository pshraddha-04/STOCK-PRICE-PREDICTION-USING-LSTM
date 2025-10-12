// Market status and timer functionality
function updateMarketStatus() {
    const now = new Date();
    const day = now.getDay();
    const hour = now.getHours();
    const minute = now.getMinutes();
    
    const statusText = document.getElementById('statusText');
    const marketTimer = document.getElementById('marketTimer');
    
    // Market is open Monday-Friday 9:30 AM - 4:00 PM EST
    const isWeekday = day >= 1 && day <= 5;
    const marketOpen = hour > 9 || (hour === 9 && minute >= 30);
    const marketClose = hour < 16;
    const isMarketOpen = isWeekday && marketOpen && marketClose;
    
    if (isMarketOpen) {
        statusText.textContent = 'OPEN';
        statusText.style.color = '#22c55e';
        
        // Calculate time until market close
        const closeTime = new Date();
        closeTime.setHours(16, 0, 0, 0);
        const timeLeft = closeTime - now;
        
        if (timeLeft > 0) {
            const hours = Math.floor(timeLeft / (1000 * 60 * 60));
            const minutes = Math.floor((timeLeft % (1000 * 60 * 60)) / (1000 * 60));
            marketTimer.textContent = `Closes in ${hours}h ${minutes}m`;
        }
    } else {
        statusText.textContent = 'CLOSED';
        statusText.style.color = '#ef4444';
        
        // Calculate time until market opens
        let nextOpen = new Date();
        if (day === 0) { // Sunday
            nextOpen.setDate(nextOpen.getDate() + 1);
        } else if (day === 6) { // Saturday
            nextOpen.setDate(nextOpen.getDate() + 2);
        } else if (!marketOpen) { // Before market opens
            // Market opens today
        } else { // After market closes
            nextOpen.setDate(nextOpen.getDate() + 1);
        }
        
        nextOpen.setHours(9, 30, 0, 0);
        const timeUntilOpen = nextOpen - now;
        
        if (timeUntilOpen > 0) {
            const hours = Math.floor(timeUntilOpen / (1000 * 60 * 60));
            const minutes = Math.floor((timeUntilOpen % (1000 * 60 * 60)) / (1000 * 60));
            marketTimer.textContent = `Opens in ${hours}h ${minutes}m`;
        }
    }
}

// Quick predict functionality
function quickPredict(symbol) {
    // Store the symbol and redirect to prediction page
    sessionStorage.setItem('quickPredictSymbol', symbol);
    window.location.href = 'index.html';
}

// Fetch live market data with improved error handling
function fetchLiveMarketData() {
    fetch('/market-data', {
        method: 'GET',
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        },
        credentials: 'same-origin'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success && data.data) {
            if (data.data.stocks) updateStockTicker(data.data.stocks);
            if (data.data.indices) updateMarketIndices(data.data.indices);
        } else {
            console.warn('Market data fetch succeeded but no data received');
        }
    })
    .catch(error => {
        console.error('Error fetching market data:', error);
        // Optionally show user-friendly error message
        const statusElement = document.getElementById('statusText');
        if (statusElement) {
            statusElement.textContent = 'DATA ERROR';
            statusElement.style.color = '#ef4444';
        }
    });
}

// Update stock ticker with real data
function updateStockTicker(stocks) {
    const tickerItems = document.querySelectorAll('.ticker-item');
    
    stocks.forEach((stock, index) => {
        if (tickerItems[index]) {
            const priceElement = tickerItems[index].querySelector('.price');
            const changeElement = tickerItems[index].querySelector('.change');
            
            priceElement.textContent = `$${stock.price}`;
            changeElement.textContent = `${stock.change >= 0 ? '+' : ''}${stock.change}%`;
            changeElement.className = `change ${stock.change >= 0 ? 'positive' : 'negative'}`;
        }
    });
}

// Update market indices with real data
function updateMarketIndices(indices) {
    const indexItems = document.querySelectorAll('.index-item');
    
    indices.forEach((index, i) => {
        if (indexItems[i]) {
            const valueElement = indexItems[i].querySelector('.index-value');
            const changeElement = indexItems[i].querySelector('.index-change');
            
            valueElement.textContent = index.value.toLocaleString();
            changeElement.textContent = `${index.change >= 0 ? '+' : ''}${index.change}%`;
            changeElement.className = `index-change ${index.change >= 0 ? 'positive' : 'negative'}`;
        }
    });
}

// Auto-fill symbol if coming from quick predict
document.addEventListener('DOMContentLoaded', function() {
    const quickSymbol = sessionStorage.getItem('quickPredictSymbol');
    if (quickSymbol && document.getElementById('stockSymbol')) {
        document.getElementById('stockSymbol').value = quickSymbol;
        sessionStorage.removeItem('quickPredictSymbol');
    }
    
    // Update market status immediately and then every minute
    updateMarketStatus();
    setInterval(updateMarketStatus, 60000);
    
    // Fetch live market data immediately and then every 5 minutes
    fetchLiveMarketData();
    setInterval(fetchLiveMarketData, 300000);
});

