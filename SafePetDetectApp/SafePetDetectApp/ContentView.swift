import SwiftUI
import CoreML
import PhotosUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var classificationLabel: String = "Classification results will be displayed here"
    @State private var isLoading: Bool = false
    @State private var dumpURL: URL? = nil

    // PyTorch mean/std for normalization
    let mean: [Float] = [0.485, 0.456, 0.406]
    let std:  [Float] = [0.229, 0.224, 0.225]

    // Load the CoreML model
    let model: pet_classifier? = {
        do {
            return try pet_classifier(configuration: MLModelConfiguration())
        } catch {
            print("Failed to load model: \(error)")
            return nil
        }
    }()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                } else {
                    Rectangle()
                        .fill(Color.secondary)
                        .frame(height: 300)
                        .overlay(Text("Select an image").foregroundColor(.white))
                }
                
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Text("Select Image")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
                .onChange(of: selectedItem) { oldItem, newItem in
                    Task {
                        if let newItem = newItem {
                            await loadImage(from: newItem)
                        }
                    }
                }
                
                if isLoading {
                    ProgressView("Classifying...")
                }
                
                Text(classificationLabel)
                    .multilineTextAlignment(.center)
                    .padding()
                
                if let url = dumpURL {
                    Text("Pixel data saved to: \(url.lastPathComponent)")
                        .foregroundColor(.blue)
                        .onTapGesture {
                            // Open CSV file in share sheet
                            let activityVC = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                            UIApplication.shared.windows.first?.rootViewController?
                                .present(activityVC, animated: true, completion: nil)
                        }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Pet Classifier")
        }
    }
    
    /// Convert selected photo to UIImage
    func loadImage(from item: PhotosPickerItem) async {
        self.isLoading = true
        self.classificationLabel = "Classifying..."
        
        if let data = try? await item.loadTransferable(type: Data.self),
           let uiImage = UIImage(data: data) {
            self.selectedImage = uiImage
            classifyAndDumpImage(uiImage)
        } else {
            self.classificationLabel = "Failed to load image"
            self.isLoading = false
        }
    }
    
    /// Preprocess and classify the image
    func classifyAndDumpImage(_ uiImage: UIImage) {
        let width = 224
        let height = 224
        
        guard let cgImage = uiImage.cgImage else {
            self.classificationLabel = "Failed to get CGImage"
            self.isLoading = false
            return
        }
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            self.classificationLabel = "Failed to create sRGB color space"
            self.isLoading = false
            return
        }
        
        // Create CGContext for resizing
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            self.classificationLabel = "Failed to create CGContext"
            self.isLoading = false
            return
        }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let resizedData = context.data else {
            self.classificationLabel = "Failed to get context data"
            self.isLoading = false
            return
        }
        
        // Create MLMultiArray for the model
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            self.classificationLabel = "Failed to create MLMultiArray"
            self.isLoading = false
            return
        }
        
        // Normalize pixel values and write to MLMultiArray
        var csvString = ""
        
        func putValue(_ batch: Int, _ channel: Int, _ y: Int, _ x: Int, _ val: Float) {
            let idx = [NSNumber(value: batch),
                       NSNumber(value: channel),
                       NSNumber(value: y),
                       NSNumber(value: x)]
            multiArray[idx] = NSNumber(value: val)
        }
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let rVal = resizedData.load(fromByteOffset: offset+0, as: UInt8.self)
                let gVal = resizedData.load(fromByteOffset: offset+1, as: UInt8.self)
                let bVal = resizedData.load(fromByteOffset: offset+2, as: UInt8.self)
                
                let rFloat = Float(rVal) / 255.0
                let gFloat = Float(gVal) / 255.0
                let bFloat = Float(bVal) / 255.0
                
                let rNorm = (rFloat - mean[0]) / std[0]
                let gNorm = (gFloat - mean[1]) / std[1]
                let bNorm = (bFloat - mean[2]) / std[2]
                
                putValue(0, 0, y, x, rNorm) // R
                putValue(0, 1, y, x, gNorm) // G
                putValue(0, 2, y, x, bNorm) // B
                
                csvString += "\(rNorm),\(gNorm),\(bNorm)\n"
            }
        }
        
        // Save CSV for debugging
        let fileName = "pixel_dump.csv"
        if let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = docDir.appendingPathComponent(fileName)
            do {
                try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
                self.dumpURL = fileURL
                print("Pixel data saved to \(fileURL)")
            } catch {
                print("Failed to save pixel data: \(error)")
            }
        }
        
        // Perform model inference
        guard let model = self.model else {
            self.classificationLabel = "Failed to load CoreML model"
            self.isLoading = false
            return
        }
        
        do {
            let output = try model.prediction(input_tensor: multiArray)
            let predClass = output.classLabel
            let probs = output.classLabel_probs
            
            if let confidence = probs[predClass] {
                let confidencePct = confidence * 100.0
                DispatchQueue.main.async {
                    self.isLoading = false
                    self.classificationLabel = "\(predClass) (\(String(format: "%.2f", confidencePct))%)"
                }
            } else {
                DispatchQueue.main.async {
                    self.isLoading = false
                    self.classificationLabel = "\(predClass) (?)"
                }
            }
        } catch {
            self.classificationLabel = "Model prediction failed: \(error.localizedDescription)"
            self.isLoading = false
        }
    }
}
