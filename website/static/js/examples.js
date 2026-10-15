"use strict";

const exampleDialog = document.getElementById("example-dialog");
const exampleContent = document.getElementById("example-content");
const palette = ["#557fa5", "#b31c38", "#658c62"];
let selectedExample;
let masterVideo;
let auxiliaryVideos = [];
let redrawCharts = [];
let previousFocus;
let animation;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let previewsPaused = reducedMotion.matches;

function updatePreviews() {
    document.querySelectorAll('#vista-overview video, .example-card video').forEach(video => {
        video.autoplay = !previewsPaused;
        if (previewsPaused || exampleDialog.open) video.pause();
        else video.play().catch(() => {});
    });
}

reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) previewsPaused = true;
    updatePreviews();
});
updatePreviews();

function node(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
}

function videoFor(sample, name, controls = false) {
    const video = node("video");
    video.src = `examples/${sample.clip_id}/${name}.mp4`;
    video.muted = true;
    video.playsInline = true;
    video.controls = controls;
    video.preload = "metadata";
    video.setAttribute("aria-label", `${sample.clip_id} ${name.replaceAll("_", " ")} video`);
    if (name === "rgb") video.poster = `examples/${sample.clip_id}/poster.jpg`;
    return video;
}

function chart(parent, title, values, duration, path = false, labels = []) {
    parent.append(node("h4", "", title));
    const canvas = node("canvas", "sensor-chart");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${title}: recorded signal with a moving playback cursor`);
    parent.append(canvas);
    if (labels.length) {
        const legend = node("p", "axis-legend");
        labels.forEach((label, index) => legend.append(node("span", ["blue", "red", "green"][index], label)));
        parent.append(legend);
    }
    return time => {
        const width = Math.max(150, canvas.clientWidth);
        const height = canvas.clientHeight || 125;
        canvas.width = width * devicePixelRatio;
        canvas.height = height * devicePixelRatio;
        const context = canvas.getContext("2d");
        context.scale(devicePixelRatio, devicePixelRatio);
        context.font = "10px Arial";
        context.fillStyle = "#fff";
        context.fillRect(0, 0, width, height);
        if (!values?.length) { context.fillStyle = "#555"; context.fillText("No recorded samples", 10, 30); return; }
        const left = 38, right = width - 12, top = 12, bottom = height - 28;
        const all = values.flatMap(row => path ? [row[2]] : row.slice(1));
        let minimum = Math.min(...all), maximum = Math.max(...all);
        const padding = (maximum - minimum || 1) * .08;
        minimum -= padding; maximum += padding;
        const horizontal = path ? values.map(row => row[1]) : [0, duration];
        let xMinimum = Math.min(...horizontal), xMaximum = Math.max(...horizontal);
        if (xMinimum === xMaximum) { xMinimum -= .001; xMaximum += .001; }
        const positionX = value => left + (value - xMinimum) / (xMaximum - xMinimum) * (right - left);
        const positionY = value => bottom - (value - minimum) / (maximum - minimum) * (bottom - top);
        context.strokeStyle = "#e8ebee";
        context.fillStyle = "#444";
        for (let tick = 0; tick < 3; tick++) {
            const value = minimum + (maximum - minimum) * tick / 2;
            const vertical = positionY(value);
            context.beginPath(); context.moveTo(left, vertical); context.lineTo(right, vertical); context.stroke();
            const tickLabel = Math.abs(maximum) >= 100 && maximum - minimum < 10 ? value.toFixed(2) : value.toPrecision(3);
            context.fillText(tickLabel, 0, vertical + 3);
            const xValue = xMinimum + (xMaximum - xMinimum) * tick / 2;
            context.fillText(xValue.toPrecision(3), positionX(xValue) - 8, bottom + 13);
        }
        context.fillText(path ? "x (m); y (m)" : "Time (s)", width / 2 - 20, height - 2);
        const count = path ? 1 : values[0].length - 1;
        for (let component = 0; component < count; component++) {
            context.strokeStyle = palette[component]; context.lineWidth = 1;
            context.beginPath();
            values.forEach((row, index) => {
                const horizontalValue = path ? row[1] : row[0];
                const verticalValue = path ? row[2] : row[component + 1];
                if (index === 0) context.moveTo(positionX(horizontalValue), positionY(verticalValue));
                else context.lineTo(positionX(horizontalValue), positionY(verticalValue));
            });
            context.stroke();
        }
        if (path) {
            const current = values.reduce((best, row) => Math.abs(row[0] - time) < Math.abs(best[0] - time) ? row : best, values[0]);
            context.fillStyle = "#b31c38";
            context.beginPath(); context.arc(positionX(current[1]), positionY(current[2]), 4, 0, Math.PI * 2); context.fill();
        } else {
            context.setLineDash([3, 3]); context.strokeStyle = "#555";
            context.beginPath(); context.moveTo(positionX(time), top); context.lineTo(positionX(time), bottom); context.stroke();
        }
    };
}

function updatePlayback() {
    if (!exampleDialog.open || !masterVideo) return;
    const time = masterVideo.currentTime;
    for (const video of auxiliaryVideos) {
        if (video.readyState >= 1 && Math.abs(video.currentTime - time) > .15) video.currentTime = Math.min(time, video.duration || time);
        if (masterVideo.paused && !video.paused) video.pause();
        else if (!masterVideo.paused && video.paused && !video.ended) video.play().catch(() => {});
        video.playbackRate = masterVideo.playbackRate;
    }
    redrawCharts.forEach(draw => draw(time));
    const clock = document.getElementById("example-clock");
    if (clock) clock.textContent = `t = ${time.toFixed(1)} / ${selectedExample.duration.toFixed(1)} s`;
    animation = requestAnimationFrame(updatePlayback);
}

function renderView(view) {
    const oldTime = masterVideo?.currentTime || 0;
    masterVideo?.pause(); auxiliaryVideos.forEach(video => video.pause());
    cancelAnimationFrame(animation);
    auxiliaryVideos = []; redrawCharts = [];
    exampleContent.replaceChildren();
    document.querySelectorAll(".example-tabs button").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.view === view)));
    const sample = selectedExample;
    masterVideo = videoFor(sample, "rgb", true);
    masterVideo.loop = true;
    const activeVideo = masterVideo;
    activeVideo.addEventListener("loadedmetadata", () => {
        if (activeVideo !== masterVideo || !exampleDialog.open) return;
        activeVideo.currentTime = Math.min(oldTime, activeVideo.duration);
        if (!reducedMotion.matches && !previewsPaused) activeVideo.play().catch(() => {});
    }, {once:true});
    if (view === "annotations") {
        const layout = node("div", "annotation-layout");
        layout.append(masterVideo);
        const text = node("div");
        for (const [title, entries] of [["Scene descriptions", sample.scenes.map(scene => ({description:scene}))], ["Question–answer pairs", sample.questions], ["Goals & instructions", sample.goals]]) {
            const block = node("section", "annotation-block"); block.append(node("h3", "", title));
            entries.forEach(entry => Object.entries(entry).forEach(([key, value]) => {
                const paragraph = node("p"); paragraph.append(node("strong", "", `${key[0].toUpperCase() + key.slice(1)}: `), document.createTextNode(value)); block.append(paragraph);
            })); text.append(block);
        }
        layout.append(text); exampleContent.append(layout);
    } else {
        const grid = node("div", "modality-grid");
        const columns = ["RGB video", "Audio", "IMU", "SLAM / tracking", "Eye tracking", "Environmental sensors"].map(title => {
            const column = node("section", "modality-column"); column.append(node("h3", "", title)); grid.append(column); return column;
        });
        columns[0].append(masterVideo);
        const clock = node("p", "example-clock"); clock.id = "example-clock"; columns[0].append(clock);
        const signals = sample.signals;
        redrawCharts.push(chart(columns[1], "Recorded waveform envelope", signals.audio, sample.duration));
        columns[1].append(node("h4", "", "Spectrogram (0–8 kHz)"));
        const spectrum = node("div", "spectrum-wrap");
        const image = node("img"); image.src = `examples/${sample.clip_id}/spectrogram.png`; image.alt = "Recorded audio spectrogram";
        const cursor = node("span", "spectrum-cursor"); spectrum.append(image, cursor); columns[1].append(spectrum);
        redrawCharts.push(time => {cursor.style.left = `${Math.max(0, Math.min(100, (time - sample.audio_offset) / sample.audio_duration * 100))}%`;});
        redrawCharts.push(chart(columns[2], "Acceleration (m/s²)", signals.acceleration, sample.duration, false, ["x", "y", "z"]));
        redrawCharts.push(chart(columns[2], "Angular velocity (rad/s)", signals.angular, sample.duration, false, ["x", "y", "z"]));
        columns[3].append(node("h4", "", "Stereo cameras"));
        const pair = node("div", "camera-pair");
        for (const name of ["slam_left", "slam_right"]) { const wrapper = node("div"); const video = videoFor(sample, name); auxiliaryVideos.push(video); wrapper.append(video, node("span", "", name === "slam_left" ? "Left" : "Right")); pair.append(wrapper); }
        columns[3].append(pair);
        redrawCharts.push(chart(columns[3], "Closed-loop trajectory (z omitted)", signals.trajectory, sample.duration, true));
        redrawCharts.push(chart(columns[3], "Relative orientation (°)", signals.orientation, sample.duration, false, ["Roll", "Pitch", "Yaw"]));
        columns[4].append(node("h4", "", "Eye cameras"));
        const eye = videoFor(sample, "eye"); auxiliaryVideos.push(eye); columns[4].append(eye);
        redrawCharts.push(chart(columns[4], "Gaze angles (°)", signals.gaze, sample.duration, false, ["Yaw", "Pitch"]));
        redrawCharts.push(chart(columns[5], "Pressure (hPa)", signals.pressure, sample.duration));
        redrawCharts.push(chart(columns[5], "Magnetic field (recorded units)", signals.magnetic, sample.duration, false, ["x", "y", "z"]));
        redrawCharts.push(chart(columns[5], "Sensor temperature (°C)", signals.temperature, sample.duration));
        exampleContent.append(grid);
    }
    animation = requestAnimationFrame(updatePlayback);
}

function openExample(sample, view, button) {
    selectedExample = sample; previousFocus = button;
    document.getElementById("example-title").textContent = `${sample.clip_id} · ${sample.category}`;
    document.querySelectorAll('#vista-overview video, .example-card video').forEach(video => video.pause());
    exampleDialog.showModal(); renderView(view);
}

document.querySelectorAll(".example-tabs button").forEach(button => button.addEventListener("click", () => renderView(button.dataset.view)));
document.getElementById("example-close").addEventListener("click", () => exampleDialog.close());
exampleDialog.addEventListener("close", () => {
    cancelAnimationFrame(animation); masterVideo?.pause(); auxiliaryVideos.forEach(video => video.pause());
    exampleContent.replaceChildren(); masterVideo = undefined;
    updatePreviews(); previousFocus?.focus();
});

fetch("examples/samples.json", {cache:'no-store'}).then(response => {
    if (!response.ok) throw new Error("Examples could not be loaded"); return response.json();
}).then(samples => {
    const grid = document.getElementById("vista-examples");
    for (const sample of samples) {
        const card = node("article", "example-card");
        const video = videoFor(sample, "rgb"); video.autoplay = !previewsPaused; video.loop = true;
        const menu = node("div", "example-menu");
        for (const view of ["modalities", "annotations"]) {
            const button = node("button", "", view[0].toUpperCase() + view.slice(1)); button.type = "button";
            button.setAttribute("aria-label", `${sample.clip_id}: ${view}`);
            button.addEventListener("click", () => openExample(sample, view, button)); menu.append(button);
        }
        const label = node("div", "example-label"); label.append(node("strong", "", sample.category), node("span", "", `${sample.clip_id} · ${sample.duration.toFixed(1)} s`));
        card.append(video, menu, label); grid.append(card);
    }
    document.getElementById("examples-status").textContent = "";
    updatePreviews();
}).catch(error => { document.getElementById("examples-status").textContent = error.message; });
