document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const videoInput = document.getElementById('video-input');
    const resultVideo = document.getElementById('result-video');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = videoInput.files[0];
        if (!file) return alert('Select a video file!');
        // Ограничение размера файла (например, 200 МБ)
        if (file.size > 200 * 1024 * 1024) {
            alert('Слишком большой файл! Максимум 200 МБ.');
            return;
        }
        const formData = new FormData();
        formData.append('video', file);

        resultVideo.src = '';
        resultVideo.poster = '';
        resultVideo.style.display = 'none';

        // Показываем индикатор загрузки
        const loading = document.createElement('div');
        loading.id = 'loading-indicator';
        loading.innerText = 'Обработка видео... Пожалуйста, подождите.';
        loading.style.margin = '20px';
        loading.style.fontWeight = 'bold';
        form.parentNode.insertBefore(loading, form.nextSibling);

        const res = await fetch('/process_video', {
            method: 'POST',
            body: formData
        });

        // Убираем индикатор загрузки
        if (loading) loading.remove();

        if (!res.ok) return alert('Processing failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        resultVideo.src = url;
        resultVideo.style.display = 'block';
    });
});