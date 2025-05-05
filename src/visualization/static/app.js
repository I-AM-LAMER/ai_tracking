document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const videoInput = document.getElementById('video-input');
    const resultVideo = document.getElementById('result-video');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = videoInput.files[0];
        if (!file) return alert('Select a video file!');
        const formData = new FormData();
        formData.append('video', file);

        resultVideo.src = '';
        resultVideo.poster = '';
        resultVideo.style.display = 'none';

        const res = await fetch('/process_video', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) return alert('Processing failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        resultVideo.src = url;
        resultVideo.style.display = 'block';
    });
});