import UIKit
import CoreML

class ViewController: UIViewController {
  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .white
    let label = UILabel(frame: view.bounds)
    label.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    label.numberOfLines = 0
    label.text = runOnce()
    view.addSubview(label)
  }

  func runOnce() -> String {
    // Pseudo-code: wire Core ML MLProgram decode-step with KV streaming.
    // Add your compiled .mlmodel to the app target and replace the placeholder below.
    // This sample returns a placeholder message to confirm the app wiring.
    return "Sample iOS skeleton ready. Add your MLModel and KV streaming loop."
  }
}


