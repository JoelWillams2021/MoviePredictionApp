// app.js
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('movieForm');
  const resultDiv = document.getElementById('predictionResult');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Hide & clear previous results
    resultDiv.classList.add('d-none');
    resultDiv.innerHTML = '';

    // 1) Collect all form inputs
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    // 2) Derive release_month from release_date if present
    if (payload.release_date) {
      const [year, month] = payload.release_date.split('-');
      payload.release_month = new Date(year, month - 1)
        .toLocaleString('default', { month: 'long' });
    }

    // 3) Derive primary_genre from the single genre input
    if (payload.genre) {
      payload.primary_genre = payload.genre.split(',')[0].trim();
    }

    console.log('🛰️  Payload:', payload);

    try {
      // 4) Send POST to your backend
        const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      // 5) Parse JSON response
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || JSON.stringify(data));
      }

      console.log('✅ Prediction response:', data);

      // 6) Render successful predictions
      resultDiv.innerHTML = `
        <div class="card">
          <div class="card-body">
            <h5 class="card-title">Prediction Results</h5>
            <p><strong>Box Office:</strong> $${Number(data.predicted_box_office).toLocaleString()}</p>
            <p><strong>Critic Score:</strong> ${Number(data.predicted_critic_score).toFixed(1)}%</p>
            <p><strong>Oscar Wins:</strong> ${data.predicted_oscar_wins}</p>
            <p><strong>Final Verdict</strong> ${data.verdict}</p>
          </div>
        </div>
      `;
    } catch (err) {
      console.error('❌ Error:', err);
      // 7) Render error message
      resultDiv.innerHTML = `
        <div class="alert alert-danger" role="alert">
          <strong>Error:</strong> ${err.message}
        </div>
      `;
    } finally {
      // 8) Show result container
      resultDiv.classList.remove('d-none');
    }
  });
});
