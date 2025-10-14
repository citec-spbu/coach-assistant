class TimePicker {
    constructor(elementId, options = {}) {
        this.element = document.getElementById(elementId);
        this.title = options.name || "";
        this.value = options.defaultValue || '00:00:00.000';
        this.onChange = options.onChange || (() => {});
        this.render();
    }
    render() {
        this.element.innerHTML = `
          <div class="timepicker-time-inputs">
            <span>${this.title}</span>
            <input type="number" id="${this.element.id}-hours" min="0" max="23" placeholder="HH" />
            <span>:</span>
            <input type="number" id="${this.element.id}-minutes" min="0" max="59" placeholder="MM" />
            <span>:</span>
            <input type="number" id="${this.element.id}-seconds" min="0" max="59" placeholder="SS" />
            <span>.</span>
            <input type="number" id="${this.element.id}-milliseconds" min="0" max="999" placeholder="MS"/>
          </div>
        `;

        this.inputs = {
            hours: document.getElementById(`${this.element.id}-hours`),
            minutes: document.getElementById(`${this.element.id}-minutes`),
            seconds: document.getElementById(`${this.element.id}-seconds`),
            milliseconds: document.getElementById(`${this.element.id}-milliseconds`)
        };
        Object.values(this.inputs).forEach(input => {
            input.addEventListener('input', () => this.updateTime());
        });
        this.parseTime(this.value);
    }
    parseTime(timeString) {
        const match = timeString.match(/(\d{2}):(\d{2}):(\d{2})\.(\d{3})/);
        if (match) {
            this.inputs.hours.value = parseInt(match[1]);
            this.inputs.minutes.value = parseInt(match[2]);
            this.inputs.seconds.value = parseInt(match[3]);
            this.inputs.milliseconds.value = parseInt(match[4]);
        }
    }
    updateTime() {
        let h = String(this.inputs.hours.value || 0).padStart(2, '0');
        let m = String(this.inputs.minutes.value || 0).padStart(2, '0');
        let s = String(this.inputs.seconds.value || 0).padStart(2, '0');
        let ms = String(this.inputs.milliseconds.value || 0).padStart(3, '0');

        this.value = `${h}:${m}:${s}.${ms}`;
        this.onChange(this.value);
    }
}
