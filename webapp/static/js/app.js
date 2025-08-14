class AirplaneDetectionApp {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.startTime = null;
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.debugBtn = document.getElementById('debugBtn');
        this.rawBtn = document.getElementById('rawBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultCanvas = document.getElementById('resultCanvas');
        this.errorMessage = document.getElementById('errorMessage');
        this.imageSize = document.getElementById('imageSize');
        this.detectionCount = document.getElementById('detectionCount');
        this.processingTime = document.getElementById('processingTime');
        this.detectionsContainer = document.getElementById('detectionsContainer');
        this.debugInfo = document.getElementById('debugInfo');
        this.debugContainer = document.getElementById('debugContainer');

        // Store current detections for filtering
        this.currentDetections = [];
        this.currentImageSize = null;

        // Confidence controls will be initialized when results are shown
        this.showHigh = null;
        this.showMedium = null;
        this.showLow = null;
    }

    setupEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => {
            this.imageInput.click();
        });

        this.imageInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFileSelect(file);
            }
        });

        // Upload button
        this.uploadBtn.addEventListener('click', () => {
            if (this.currentImage) {
                this.processImage();
            }
        });

        // Debug buttons
        this.debugBtn.addEventListener('click', () => {
            if (this.currentImage) {
                this.processImage('/debug-predict');
            }
        });

        this.rawBtn?.addEventListener('click', () => {
            if (this.currentImage) {
                this.processImage('/debug-raw');
            }
        });

        // Confidence control listeners will be set up when results are shown
    }

    handleFileSelect(file) {
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showError('File size must be less than 10MB');
            return;
        }

        this.currentImage = file;
        this.updateUploadButton(true);
        this.hideError();
        this.hideResults();

        // Preview the image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.displayPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    displayPreview(imageSrc) {
        const img = new Image();
        img.onload = () => {
            this.setupCanvas(img);
            this.drawImage(img);
        };
        img.src = imageSrc;
    }

    setupCanvas(img) {
        this.canvas = this.resultCanvas;
        this.ctx = this.canvas.getContext('2d');

        // Set canvas size to match image aspect ratio
        const maxWidth = 800;
        const maxHeight = 600;
        let { width, height } = img;

        if (width > maxWidth) {
            height = (height * maxWidth) / width;
            width = maxWidth;
        }

        if (height > maxHeight) {
            width = (width * maxHeight) / height;
            height = maxHeight;
        }

        this.canvas.width = width;
        this.canvas.height = height;
    }

    drawImage(img) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    }

    async processImage(endpoint = '/predict') {
        if (!this.currentImage) return;

        this.startTime = Date.now();
        this.setLoading(true, endpoint);
        this.hideError();

        const formData = new FormData();
        formData.append('image', this.currentImage);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Processing failed');
            }

            if (endpoint === '/debug-raw') {
                this.displayRawResults(result);
            } else {
                this.displayResults(result);
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showError(error.message || 'Failed to process image');
        } finally {
            console.log('Finally block executed, calling setLoading(false)');
            this.setLoading(false);
        }
    }

    displayRawResults(result) {
        const processingTimeMs = Date.now() - this.startTime;

        // Show debug info section
        this.debugInfo.hidden = false;
        this.debugContainer.innerHTML = '';

        if (result.statistics) {
            const stats = result.statistics;

            // Raw confidence stats
            const rawConfDiv = document.createElement('div');
            rawConfDiv.className = 'debug-item';
            rawConfDiv.innerHTML = `
                <h4>Confidence Statistics (Raw Values)</h4>
                <div class="debug-stat">
                    <span class="debug-label">Min:</span>
                    <span class="debug-value">${stats.raw_confidence.min.toFixed(6)}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Max:</span>
                    <span class="debug-value">${stats.raw_confidence.max.toFixed(6)}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Mean:</span>
                    <span class="debug-value">${stats.raw_confidence.mean.toFixed(6)}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Std Dev:</span>
                    <span class="debug-value">${stats.raw_confidence.std.toFixed(6)}</span>
                </div>
            `;
            this.debugContainer.appendChild(rawConfDiv);

            // Threshold analysis
            const thresholdDiv = document.createElement('div');
            thresholdDiv.className = 'debug-item';
            thresholdDiv.innerHTML = `
                <h4>Threshold Analysis (Raw Values)</h4>
                <div class="debug-stat">
                    <span class="debug-label">Above 0.001:</span>
                    <span class="debug-value">${stats.threshold_analysis.above_0_001} / ${stats.threshold_analysis.total_predictions}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Above 0.005:</span>
                    <span class="debug-value">${stats.threshold_analysis.above_0_005} / ${stats.threshold_analysis.total_predictions}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Above 0.01:</span>
                    <span class="debug-value">${stats.threshold_analysis.above_0_01} / ${stats.threshold_analysis.total_predictions}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Above 0.02:</span>
                    <span class="debug-value">${stats.threshold_analysis.above_0_02} / ${stats.threshold_analysis.total_predictions}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Above 0.05:</span>
                    <span class="debug-value">${stats.threshold_analysis.above_0_05} / ${stats.threshold_analysis.total_predictions}</span>
                </div>
            `;
            this.debugContainer.appendChild(thresholdDiv);
        }

        // Processing time
        const processingDiv = document.createElement('div');
        processingDiv.className = 'debug-item';
        processingDiv.innerHTML = `
            <h4>Processing Info</h4>
            <div class="debug-stat">
                <span class="debug-label">Processing Time:</span>
                <span class="debug-value">${processingTimeMs}ms</span>
            </div>
            <div class="debug-stat">
                <span class="debug-label">Tile Location:</span>
                <span class="debug-value">[${result.tile_location.join(', ')}]</span>
            </div>
        `;
        this.debugContainer.appendChild(processingDiv);

        this.debugInfo.classList.add('fade-in');
    }

    setLoading(loading, endpoint = '/predict') {
        const buttons = [this.uploadBtn, this.debugBtn, this.rawBtn].filter(btn => btn !== null);

        console.log('setLoading called:', { loading, endpoint, buttonsFound: buttons.length });

        buttons.forEach(btn => {
            const btnText = btn.querySelector('.btn-text');
            const btnLoader = btn.querySelector('.btn-loader');

            if (!btnText || !btnLoader) {
                console.warn('Button elements not found for:', btn.id);
                return;
            }

            if (loading) {
                btnText.hidden = true;
                btnLoader.hidden = false;
                btn.disabled = true;
                console.log('Set button to loading state:', btn.id);
            } else {
                btnText.hidden = false;
                btnLoader.hidden = true;
                btn.disabled = !this.currentImage; // Only enable if image is selected
                console.log('Reset button from loading state:', btn.id, 'disabled:', btn.disabled);
            }
        });

        // Special handling for main upload button text
        if (!loading && this.uploadBtn) {
            const mainBtnText = this.uploadBtn.querySelector('.btn-text');
            if (mainBtnText && this.currentImage) {
                mainBtnText.textContent = 'Detect Airplanes';
            }
        }
    }

    filterDetections() {
        console.log('filterDetections called');
        console.log('Current detections:', this.currentDetections?.length);
        console.log('Current image loaded:', !!this.currentImage_loaded);

        if (!this.currentDetections || this.currentDetections.length === 0 || !this.currentImage_loaded) {
            console.log('Early return from filterDetections');
            return;
        }

        const showHigh = this.showHigh?.checked !== false;
        const showMedium = this.showMedium?.checked !== false;
        const showLow = this.showLow?.checked !== false;

        console.log('Filter settings:', { showHigh, showMedium, showLow });

        // Clear canvas and redraw from original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawImage(this.currentImage_loaded);

        // Draw only the filtered detections
        this.drawFilteredDetections(this.currentDetections, this.currentImageSize, showHigh, showMedium, showLow);

        // Update the detection count display
        const visibleCount = this.countVisibleDetections(this.currentDetections, showHigh, showMedium, showLow);
        console.log('Visible count:', visibleCount, 'of', this.currentDetections.length);
        this.updateDetectionCount(visibleCount, this.currentDetections.length);
    }

    countVisibleDetections(detections, showHigh, showMedium, showLow) {
        return detections.filter(detection => {
            if (detection.confidence >= 0.7 && showHigh) return true;
            if (detection.confidence >= 0.5 && detection.confidence < 0.7 && showMedium) return true;
            if (detection.confidence >= 0.25 && detection.confidence < 0.5 && showLow) return true;
            return false;
        }).length;
    }

    updateDetectionCount(visibleCount, totalCount) {
        if (visibleCount === totalCount) {
            this.detectionCount.textContent = `${totalCount} total (H:${this.currentDetections.filter(d => d.confidence >= 0.7).length}, M:${this.currentDetections.filter(d => d.confidence >= 0.5 && d.confidence < 0.7).length}, L:${this.currentDetections.filter(d => d.confidence >= 0.25 && d.confidence < 0.5).length})`;
        } else {
            this.detectionCount.textContent = `${visibleCount} visible / ${totalCount} total`;
        }
    }

    drawFilteredDetections(detections, originalSize, showHigh, showMedium, showLow) {
        const scaleX = this.canvas.width / originalSize[0];
        const scaleY = this.canvas.height / originalSize[1];

        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;

            // Check confidence level and whether to show this detection
            let shouldShow = false;
            if (detection.confidence >= 0.7 && showHigh) shouldShow = true;
            else if (detection.confidence >= 0.5 && detection.confidence < 0.7 && showMedium) shouldShow = true;
            else if (detection.confidence >= 0.25 && detection.confidence < 0.5 && showLow) shouldShow = true;

            if (!shouldShow) return;

            // Scale coordinates to canvas size
            const canvasX1 = x1 * scaleX;
            const canvasY1 = y1 * scaleY;
            const canvasX2 = x2 * scaleX;
            const canvasY2 = y2 * scaleY;

            const width = canvasX2 - canvasX1;
            const height = canvasY2 - canvasY1;

            // Choose color based on confidence level
            let strokeColor, fillColor, confLevel;
            if (detection.confidence >= 0.7) {
                strokeColor = '#ff4444';
                fillColor = '#ff4444';
                confLevel = 'HIGH';
            } else if (detection.confidence >= 0.5) {
                strokeColor = '#ff8800';
                fillColor = '#ff8800';
                confLevel = 'MED';
            } else {
                strokeColor = '#ffaa00';
                fillColor = '#ffaa00';
                confLevel = 'LOW';
            }

            // Draw bounding box with confidence-based color (no labels)
            this.ctx.strokeStyle = strokeColor;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(canvasX1, canvasY1, width, height);
        });
    }

    displayResults(result) {
        const { image, detections, image_size, confidence_breakdown, tile_config } = result;
        const processingTimeMs = Date.now() - this.startTime;

        // Store current detections and image for filtering
        this.currentDetections = detections;
        this.currentImageSize = image_size;

        // Handle debug mode
        if (result.debug_mode) {
            this.displayDebugInfo(result);
        } else {
            if (this.debugInfo) this.debugInfo.hidden = true;
        }

        // Load and display the image
        const img = new Image();
        img.onload = () => {
            // Store the original image for filtering
            this.currentImage_loaded = img;

            this.setupCanvas(img);
            this.drawImage(img);
            this.drawDetections(detections, image_size);
            this.updateResultsInfo(image_size, detections.length, processingTimeMs, confidence_breakdown, tile_config);
            this.displayDetectionsList(detections);
            this.showResults();
        };
        img.src = image;
    }

    displayDebugInfo(result) {
        this.debugInfo.hidden = false;
        this.debugContainer.innerHTML = '';

        if (result.confidence_stats) {
            const statsDiv = document.createElement('div');
            statsDiv.className = 'debug-item';
            statsDiv.innerHTML = `
                <h4>Debug Statistics</h4>
                <div class="debug-stat">
                    <span class="debug-label">Detection Count:</span>
                    <span class="debug-value">${result.detection_count}</span>
                </div>
                <div class="debug-stat">
                    <span class="debug-label">Confidence Threshold:</span>
                    <span class="debug-value">${result.confidence_threshold}</span>
                </div>
            `;

            if (result.confidence_stats.min !== undefined) {
                statsDiv.innerHTML += `
                    <div class="debug-stat">
                        <span class="debug-label">Confidence Range:</span>
                        <span class="debug-value">${result.confidence_stats.min.toFixed(3)} - ${result.confidence_stats.max.toFixed(3)}</span>
                    </div>
                    <div class="debug-stat">
                        <span class="debug-label">Average Confidence:</span>
                        <span class="debug-value">${result.confidence_stats.mean.toFixed(3)}</span>
                    </div>
                `;
            }

            this.debugContainer.appendChild(statsDiv);
        }

        this.debugInfo.classList.add('fade-in');
    }

    drawDetections(detections, originalSize) {
        const scaleX = this.canvas.width / originalSize[0];
        const scaleY = this.canvas.height / originalSize[1];

        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;

            // Scale coordinates to canvas size
            const canvasX1 = x1 * scaleX;
            const canvasY1 = y1 * scaleY;
            const canvasX2 = x2 * scaleX;
            const canvasY2 = y2 * scaleY;

            const width = canvasX2 - canvasX1;
            const height = canvasY2 - canvasY1;

            // Choose color based on confidence level (matching notebook visualization)
            let strokeColor, fillColor, confLevel;
            if (detection.confidence >= 0.7) {
                strokeColor = '#ff4444'; // Red for high confidence
                fillColor = '#ff4444';
                confLevel = 'HIGH';
            } else if (detection.confidence >= 0.5) {
                strokeColor = '#ff8800'; // Orange for medium confidence
                fillColor = '#ff8800';
                confLevel = 'MED';
            } else {
                strokeColor = '#ffdd00'; // Yellow for low confidence
                fillColor = '#ffaa00';
                confLevel = 'LOW';
            }

            // Draw bounding box with confidence-based color (no labels)
            this.ctx.strokeStyle = strokeColor;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(canvasX1, canvasY1, width, height);
        });
    }

    displayDetectionsList(detections) {
        this.detectionsContainer.innerHTML = '';

        if (detections.length === 0) {
            this.detectionsContainer.innerHTML = '<p style="text-align: center; color: #666;">No aircraft detected</p>';
            return;
        }

        // Sort detections by confidence (highest first)
        const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);

        sortedDetections.forEach((detection, index) => {
            const detectionElement = document.createElement('div');
            detectionElement.className = 'detection-item';

            const [x1, y1, x2, y2] = detection.bbox;

            // Determine confidence level and color
            let confidenceClass, confidenceLevel;
            if (detection.confidence >= 0.7) {
                confidenceClass = 'confidence-high';
                confidenceLevel = 'HIGH';
            } else if (detection.confidence >= 0.5) {
                confidenceClass = 'confidence-medium';
                confidenceLevel = 'MED';
            } else {
                confidenceClass = 'confidence-low';
                confidenceLevel = 'LOW';
            }

            detectionElement.classList.add(confidenceClass);

            detectionElement.innerHTML = `
                <div class="detection-class">${detection.class} #${index + 1}</div>
                <div class="detection-confidence">
                    <span class="confidence-badge ${confidenceClass}">${confidenceLevel}</span>
                    ${(detection.confidence * 100).toFixed(1)}% confidence
                </div>
                <div class="detection-bbox">
                    Location: (${Math.round(x1)}, ${Math.round(y1)}) → (${Math.round(x2)}, ${Math.round(y2)})
                </div>
                <div class="detection-size">
                    Size: ${Math.round(x2-x1)} × ${Math.round(y2-y1)} pixels
                </div>
            `;

            this.detectionsContainer.appendChild(detectionElement);
        });
    }

    updateResultsInfo(imageSize, detectionCount, processingTime, confidenceBreakdown, tileConfig) {
        this.imageSize.textContent = `${imageSize[0]} × ${imageSize[1]}`;

        // Enhanced detection count with confidence breakdown
        if (confidenceBreakdown) {
            const breakdown = `${detectionCount} total (H:${confidenceBreakdown.high}, M:${confidenceBreakdown.medium}, L:${confidenceBreakdown.low})`;
            this.detectionCount.textContent = breakdown;
            this.detectionCount.title = 'High ≥0.7, Medium 0.5-0.7, Low 0.25-0.5';
        } else {
            this.detectionCount.textContent = detectionCount;
        }

        // Enhanced processing time with inference method
        let timeText = `${processingTime}ms`;
        if (tileConfig) {
            timeText += ` (Tiled: ${tileConfig.tile_size}, overlap: ${tileConfig.overlap}px)`;
        }
        this.processingTime.textContent = timeText;
    }

    updateUploadButton(hasFile) {
        const btnText = this.uploadBtn.querySelector('.btn-text');
        if (hasFile) {
            btnText.textContent = 'Detect Airplanes';
            this.uploadBtn.disabled = false;
            if (this.debugBtn) this.debugBtn.disabled = false;
            if (this.rawBtn) this.rawBtn.disabled = false;
        } else {
            btnText.textContent = 'Select Image First';
            this.uploadBtn.disabled = true;
            if (this.debugBtn) this.debugBtn.disabled = true;
            if (this.rawBtn) this.rawBtn.disabled = true;
        }
    }

    setLoading(loading) {
        const btnText = this.uploadBtn.querySelector('.btn-text');
        const btnLoader = this.uploadBtn.querySelector('.btn-loader');

        if (loading) {
            btnText.hidden = true;
            btnLoader.hidden = false;
            this.uploadBtn.disabled = true;
        } else {
            btnText.hidden = false;
            btnLoader.hidden = true;
            this.uploadBtn.disabled = false;
        }
    }

    showResults() {
        this.resultsSection.hidden = false;
        this.resultsSection.classList.add('fade-in');

        // Now that results are shown, set up confidence control listeners
        this.setupConfidenceControls();
    }

    setupConfidenceControls() {
        // Find confidence control elements (now that they're visible)
        this.showHigh = document.getElementById('showHigh');
        this.showMedium = document.getElementById('showMedium');
        this.showLow = document.getElementById('showLow');

        console.log('Setting up confidence controls:', {
            high: !!this.showHigh,
            medium: !!this.showMedium,
            low: !!this.showLow
        });

        // Remove existing listeners first to avoid duplicates
        if (this.showHigh) {
            this.showHigh.removeEventListener('change', this.handleHighChange);
            this.handleHighChange = () => {
                console.log('High checkbox changed:', this.showHigh.checked);
                this.filterDetections();
            };
            this.showHigh.addEventListener('change', this.handleHighChange);
        }

        if (this.showMedium) {
            this.showMedium.removeEventListener('change', this.handleMediumChange);
            this.handleMediumChange = () => {
                console.log('Medium checkbox changed:', this.showMedium.checked);
                this.filterDetections();
            };
            this.showMedium.addEventListener('change', this.handleMediumChange);
        }

        if (this.showLow) {
            this.showLow.removeEventListener('change', this.handleLowChange);
            this.handleLowChange = () => {
                console.log('Low checkbox changed:', this.showLow.checked);
                this.filterDetections();
            };
            this.showLow.addEventListener('change', this.handleLowChange);
        }
    }

    hideResults() {
        this.resultsSection.hidden = true;
        this.resultsSection.classList.remove('fade-in');
    }

    showError(message) {
        const errorText = this.errorMessage.querySelector('.error-text');
        errorText.textContent = message;
        this.errorMessage.hidden = false;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.errorMessage.hidden = true;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AirplaneDetectionApp();
});
