// app.js
// script/app.js

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('movieForm');
  const resultDiv = document.getElementById('predictionResult');
  const recoInput = document.getElementById('recoGenreInput');
  const addGenreBtn = document.getElementById('addGenreBtn');
  const recoList = document.getElementById('recoList');

  // 1) Handle prediction form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultDiv.classList.add('d-none');
    resultDiv.innerHTML = '';

    // collect payload
    const data = {};
    new FormData(form).forEach((v, k) => data[k] = v);

    // derive release_month & primary_genre
    if (data.release_date && !data.release_month) {
      const [y, m] = data.release_date.split('-');
      data.release_month = new Date(y, m - 1).toLocaleString('default', { month: 'long' });
    }
    if (data.genre && !data.primary_genre) {
      data.primary_genre = data.genre.split(',')[0].trim();
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Prediction failed');

      resultDiv.innerHTML = `
        <div class="card">
          <div class="card-body">
            <h5 class="card-title">Prediction Results</h5>
            <p><strong>Box Office:</strong> $${Number(json.predicted_box_office).toLocaleString()}</p>
            <p><strong>Critic Score:</strong> ${Number(json.predicted_critic_score).toFixed(1)}%</p>
            <p><strong>Oscar Wins:</strong> ${json.predicted_oscar_wins}</p>
            <p><strong>Verdict:</strong> ${json.verdict}</p>
          </div>
        </div>
      `;
    } catch (err) {
      resultDiv.innerHTML = `
        <div class="alert alert-danger" role="alert">
          <strong>Error:</strong> ${err.message}
        </div>
      `;
    } finally {
      resultDiv.classList.remove('d-none');
    }
  });

  // 2) Handle ROI-based recommendations
  addGenreBtn.addEventListener('click', async () => {
  const genre = recoGenreInput.value.trim();
  if (!genre) return;

  // Clear previous recommendations
  recoList.innerHTML = '';

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genre }),
    });
    const suggestions = await res.json();
    if (!res.ok) throw new Error(suggestions.error || 'Recommendation failed');

    console.log('got recommendations:', suggestions);

    suggestions.forEach((movie, i) => {
      // Create card with a canvas placeholder
      const card = document.createElement('div');
      card.className = 'card mb-4 p-3';
      card.innerHTML = `
        <div class="card-body">
          <h5 class="card-title">${movie.title}</h5>
          <p><strong>Ideal Director:</strong> ${movie.director}</p>
          <p><strong>Suggested Cast:</strong> ${movie.cast.join(', ')}</p>
          <canvas id="chart-${i}" style="max-width: 100%; height: 200px;"></canvas>
        </div>
      `;
      recoList.appendChild(card);

      // Render Budget vs Gross bar chart (in millions)
      new Chart(
        document.getElementById(`chart-${i}`),
        {
          type: 'bar',
          data: {
            labels: ['Budget', 'Gross'],
            datasets: [{
              label: '$ (M)',
              data: [
                movie.budget / 1e6,
                movie.gross_worldwide / 1e6
              ],
              backgroundColor: [
                'rgba(54, 162, 235, 0.5)',
                'rgba(75, 192, 192, 0.5)'
              ]
            }]
          },
          options: {
            scales: {
              y: {
                beginAtZero: true,
                ticks: {
                  callback: v => `$${v}M`
                }
              }
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => `$${ctx.parsed.y.toFixed(2)}M`
                }
              }
            }
          }
        }
      );
    });
  } catch (err) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger';
    alert.textContent = `Error: ${err.message}`;
    recoList.appendChild(alert);
  } finally {
    recoGenreInput.value = '';
  }
});
});