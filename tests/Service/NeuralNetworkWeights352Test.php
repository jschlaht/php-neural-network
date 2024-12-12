<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkWeights352Test extends TestCase
{
    private array $weightsInputToHiddden = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6],
        [0.6, 0.3, 0.8],
        [0.7, 0.1, 0.3],
    ];
    private array $weightsInputToHidddenChangeOne = [
        [0.89223848, 0.29913761, 0.39310087],
        [0.1896559,  0.79885065, 0.19080524],
        [0.0940615,  0.49934017, 0.59472133],
        [0.59738989,  0.29970999, 0.7976799],
        [0.68689837, 0.09854426, 0.2883541 ],
    ];
    private array $deltaWeightsInputToHiddden = [
         [-0.00776152, -0.00086239, -0.00689913],
         [-0.0103441,  -0.00114935, -0.00919476],
         [-0.0059385,  -0.00065983, -0.00527867],
         [-0.00261011,  -0.00029001, -0.0023201],
         [-0.01310163, -0.00145574, -0.0116459 ],
    ];
    private array $weightsHiddenToOutput = [
        [0.3, 0.7, 0.8, 0.5, 0.4],
        [0.6, 0.5, 0.2, 0.1, 0.9],
    ];

    private array $weightsHiddenToOutputChangeOne = [
        [0.29830178, 0.69865388, 0.79854963, 0.49828178, 0.39842337],
        [0.59259432, 0.49412976, 0.19367515, 0.0925071,  0.89312455]
    ];
    private array $deltaWeightsHiddenToOutput = [
        [-0.00169822, -0.00134612, -0.00145037, -0.00171822, -0.00157663],
        [-0.00740568, -0.00587024, -0.00632485, -0.0074929,  -0.00687545],
    ];

    private array $expectedHiddenInputs = [
        [1.16],
        [0.42],
        [0.62],
        [1.21],
        [0.88],
    ];
    private array $expectedHiddenOutputs = [
        [0.76133271],
        [0.60348325],
        [0.65021855],
        [0.77029895],
        [0.70682222],
    ];
    private array $expectedFinalInputs = [
        [1.83889129],
        [1.60175485]
    ];
    private array $expectedFinalOutputs = [
        [0.86281753],
        [0.83226351]
    ];
    private array $expectedOutputErrors = [
        [-0.06281753],
        [-0.23226351]
    ];
    private array $expectedHiddenErrors = [
        [-0.15820337],
        [-0.16010403],
        [-0.09670673],
        [-0.05463512],
        [-0.23416417]
    ];
    private int $inputNodes = 3;
    private int $hiddenNodes = 5;
    private int $outputNodes = 2;
    private float $learningRate = 0.3;

    public function testTrain()
    {
        $neuralNetwork = new NeuralNetwork($this->inputNodes, $this->hiddenNodes, $this->outputNodes, $this->learningRate);

        $inputList = array(0.9, 0.1, 0.8);
        $targetList = array(0.8, 0.6);

        $neuralNetwork->setWeightsInputToHidden($this->weightsInputToHiddden);
        $neuralNetwork->setWeightsHiddenToOutput($this->weightsHiddenToOutput);

        $neuralNetwork->train($inputList, $targetList);

        $this->assertSame($this->expectedHiddenInputs, $neuralNetwork->getHiddenInputs());
        $this->assertSame($this->expectedHiddenOutputs, $neuralNetwork->getHiddenOutputs());
        $this->assertSame($this->expectedFinalInputs, $neuralNetwork->getFinalInputs());
        $this->assertSame($this->expectedFinalOutputs, $neuralNetwork->getFinalOutputs());
        $this->assertSame($this->expectedOutputErrors, $neuralNetwork->getOutputErrors());
        $this->assertSame($this->expectedHiddenErrors, $neuralNetwork->getHiddenErrors());
        $this->assertSame($this->deltaWeightsHiddenToOutput, $neuralNetwork->getDeltaWeightsHiddenToOutput());
        $this->assertSame($this->deltaWeightsInputToHiddden, $neuralNetwork->getDeltaWeightsInputToHidden());
        $this->assertSame($this->weightsHiddenToOutputChangeOne, $neuralNetwork->getWeightsHiddenToOutput());
        $this->assertSame($this->weightsInputToHidddenChangeOne, $neuralNetwork->getWeightsInputToHidden());
    }
}
