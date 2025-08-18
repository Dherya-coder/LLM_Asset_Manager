document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('investmentForm');
    const resultsSection = document.getElementById('resultsSection');
    const loading = document.getElementById('loading');
    const recommendationText = document.getElementById('recommendationText');
    let allocationChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const age = document.getElementById('age').value;
        const investmentAmount = document.getElementById('investmentAmount').value;

        // Show loading state
        loading.style.display = 'block';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/allocate-investment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    age: parseInt(age),
                    investment_amount: parseFloat(investmentAmount)
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get recommendations');
            }

            const data = await response.json();
            
            // Update recommendation text
            recommendationText.textContent = data.recommendation;

            // Update or create chart
            if (allocationChart) {
                allocationChart.destroy();
            }

            const ctx = document.getElementById('allocationChart').getContext('2d');
            allocationChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(data.allocation_percentage),
                    datasets: [{
                        data: Object.values(data.allocation_percentage),
                        backgroundColor: [
                            '#2563eb', // Blue
                            '#16a34a', // Green
                            '#dc2626', // Red
                            '#9333ea', // Purple
                            '#ea580c', // Orange
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                font: {
                                    size: 14
                                }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw || 0;
                                    return `${label}: ${value}%`;
                                }
                            }
                        }
                    }
                }
            });

            // Show results
            resultsSection.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to get investment recommendations. Please try again.');
        } finally {
            loading.style.display = 'none';
        }
    });
}); 