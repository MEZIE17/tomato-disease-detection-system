(function () {
  const dropzone = document.getElementById("dropzone");
  const input = document.getElementById("leaf_image");
  const idle = document.getElementById("dropzone-idle");
  const previewImg = document.getElementById("preview-image");
  const submitBtn = document.getElementById("submit-btn");

  if (!dropzone || !input) return;

  function showPreview(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewImg.hidden = false;
      idle.hidden = true;
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  input.addEventListener("change", function () {
    if (input.files && input.files[0]) {
      showPreview(input.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", function (e) {
    const file = e.dataTransfer.files[0];
    if (file) {
      input.files = e.dataTransfer.files;
      showPreview(file);
    }
  });

  const form = document.getElementById("upload-form");
  if (form) {
    form.addEventListener("submit", function () {
      if (submitBtn && !submitBtn.disabled) {
        submitBtn.textContent = "Diagnosing…";
        submitBtn.disabled = true;
      }
    });
  }

  // -------------------------------------------------------------------
  // Camera capture
  // -------------------------------------------------------------------
  const cameraBtn = document.getElementById("camera-btn");
  const cameraModal = document.getElementById("camera-modal");
  const cameraVideo = document.getElementById("camera-video");
  const cameraCanvas = document.getElementById("camera-canvas");
  const cameraCapture = document.getElementById("camera-capture");
  const cameraCancel = document.getElementById("camera-cancel");
  const cameraError = document.getElementById("camera-error");

  let activeStream = null;

  function stopStream() {
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop());
      activeStream = null;
    }
  }

  function closeCameraModal() {
    stopStream();
    cameraModal.hidden = true;
    cameraError.hidden = true;
    cameraVideo.hidden = false;
    cameraCapture.hidden = false;
  }

  if (cameraBtn && cameraModal) {
    cameraBtn.addEventListener("click", async function () {
      cameraModal.hidden = false;
      cameraError.hidden = true;
      cameraVideo.hidden = false;
      cameraCapture.hidden = false;

      async function attachStream(stream) {
        activeStream = stream;
        cameraVideo.srcObject = stream;
        try {
          await cameraVideo.play();
        } catch (playErr) {
          // Some browsers require a user gesture; the click that opened
          // this modal usually counts, but if not, video stays paused
          // on the first frame rather than fully failing.
          console.warn("Video play() was blocked:", playErr);
        }
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraError.hidden = false;
        cameraError.textContent =
          "Camera access isn't supported in this browser, or this page isn't served over HTTPS/localhost. Use the upload option instead.";
        cameraVideo.hidden = true;
        cameraCapture.hidden = true;
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        await attachStream(stream);
      } catch (err) {
        // Fall back to any available camera if "environment" isn't supported
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          await attachStream(stream);
        } catch (err2) {
          cameraError.hidden = false;
          cameraError.textContent =
            "Couldn't access your camera (" + (err2.name || "unknown error") +
            "). Check that no other app is using the camera, that camera permission is allowed for this site, and that Windows camera privacy settings allow browser access.";
          cameraVideo.hidden = true;
          cameraCapture.hidden = true;
        }
      }
    });

    cameraCancel.addEventListener("click", closeCameraModal);

    cameraCapture.addEventListener("click", function () {
      const width = cameraVideo.videoWidth;
      const height = cameraVideo.videoHeight;
      if (!width || !height) return;

      cameraCanvas.width = width;
      cameraCanvas.height = height;
      const ctx = cameraCanvas.getContext("2d");
      ctx.drawImage(cameraVideo, 0, 0, width, height);

      cameraCanvas.toBlob(function (blob) {
        if (!blob) return;
        const file = new File([blob], `leaf-capture-${Date.now()}.jpg`, { type: "image/jpeg" });

        // Assign the captured photo into the same file input the form submits
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;

        showPreview(file);
        closeCameraModal();
      }, "image/jpeg", 0.92);
    });

    // Close modal if user clicks the dark backdrop
    cameraModal.addEventListener("click", function (e) {
      if (e.target === cameraModal) closeCameraModal();
    });
  }
})();