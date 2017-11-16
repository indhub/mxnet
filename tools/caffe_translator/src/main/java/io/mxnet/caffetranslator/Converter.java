package io.mxnet.caffetranslator;

import io.mxnet.caffetranslator.generators.*;
import lombok.Setter;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroup;
import org.stringtemplate.v4.STRawGroupDir;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Converter {

    private final String trainPrototxt, solverPrototxt;
    private Solver solver;
    private MLModel mlModel;
    private STGroup stGroup;
    private SymbolGeneratorFactory generators;
    private final String NL;
    private GenHelper gh;
    @Setter
    private String paramsFilePath;


    Converter(String trainPrototxt, String solverPrototxt) {
        this.trainPrototxt = trainPrototxt;
        this.solverPrototxt = solverPrototxt;
        this.mlModel = new MLModel();
        this.stGroup = new STRawGroupDir("templates");
        this.generators = SymbolGeneratorFactory.getInstance();
        NL = System.getProperty("line.separator");
        gh = new GenHelper();
        addGenerators();
    }

    private void addGenerators() {
        generators.addGenerator("Convolution", new ConvolutionGenerator());
        generators.addGenerator("Deconvolution", new DeconvolutionGenerator());
        generators.addGenerator("Pooling", new PoolingGenerator());
        generators.addGenerator("InnerProduct", new FCGenerator());
        generators.addGenerator("ReLU", new ReluGenerator());
        generators.addGenerator("SoftmaxWithLoss", new SoftmaxOutputGenerator());
        generators.addGenerator("PluginIntLayerGenerator", new PluginIntLayerGenerator());
        generators.addGenerator("CaffePluginLossLayer", new PluginLossGenerator());
        generators.addGenerator("Permute", new PermuteGenerator());
        generators.addGenerator("Concat", new ConcatGenerator());
        generators.addGenerator("BatchNorm", new BatchNormGenerator());
        generators.addGenerator("Power", new PowerGenerator());
        generators.addGenerator("Eltwise", new EltwiseGenerator());
        generators.addGenerator("Flatten", new FlattenGenerator());
        generators.addGenerator("Dropout", new DropoutGenerator());
        generators.addGenerator("Scale", new ScaleGenerator());
    }

    public void parseTrainingPrototxt() {

        CharStream cs = null;
        try {
            FileInputStream fis = new FileInputStream(new File(trainPrototxt));
            cs = CharStreams.fromStream(fis, StandardCharsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }

        CaffePrototxtLexer lexer = new CaffePrototxtLexer(cs);

        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CaffePrototxtParser parser = new CaffePrototxtParser(tokens);

        CreateModelListener modelCreator = new CreateModelListener(parser, mlModel);
        parser.addParseListener(modelCreator);
        parser.prototxt();
    }

    public void parseSolverPrototxt() {
        solver = new Solver(solverPrototxt);
        solver.parsePrototxt();
    }

    public String generateMXNetCode() {
        parseTrainingPrototxt();
        parseSolverPrototxt();

        StringBuilder code = new StringBuilder();

        code.append(generateImports());
        code.append(System.lineSeparator());

        code.append(generateLogger());
        code.append(System.lineSeparator());

        code.append(generateParamInitializer());
        code.append(System.lineSeparator());

        code.append(generateMetricsClasses());
        code.append(System.lineSeparator());

        if(paramsFilePath != null) {
            code.append(generateParamsLoader());
            code.append(System.lineSeparator());
        }

        // Convert data layers
        code.append(generateIterators());

        // Generate variables for data and label
        code.append(generateInputVars());

        // Convert non data layers
        List<Layer> layers = mlModel.getNonDataLayers();

        for (int layerIndex = 0; layerIndex < layers.size(); ) {
            Layer layer = layers.get(layerIndex);
            SymbolGenerator generator = generators.getGenerator(layer.getType());

            // If the translator cannot translate this layer to an MXNet layer,
            // use CaffeOp or CaffeLoss instead.
            if (generator == null) {
                if (layer.getType().toLowerCase().endsWith("loss")) {
                    generator = generators.getGenerator("CaffePluginLossLayer");
                } else {
                    generator = generators.getGenerator("PluginIntLayerGenerator");
                }
            }

            GeneratorOutput out = generator.generate(layer, mlModel);
            String segment = out.code;
            code.append(segment);
            code.append(NL);

            layerIndex += out.numLayersTranslated;
        }

        String loss = getLoss(mlModel, code);

        String evalMetric = generateValidationMetrics(mlModel);
        code.append(evalMetric);

        String runner = generateRunner(loss);
        code.append(runner);

        return code.toString();
    }

    private String generateLogger() {
        ST st = gh.getTemplate("logging");
        st.add("name", mlModel.getName());
        return st.render();
    }

    private String generateRunner(String loss) {
        ST st = gh.getTemplate("runner");
        st.add("max_iter", solver.getProperty("max_iter"));
        st.add("stepsize", solver.getProperty("stepsize"));
        st.add("snapshot", solver.getProperty("snapshot"));
        st.add("test_interval", solver.getProperty("test_interval"));
        st.add("test_iter", solver.getProperty("test_iter"));
        st.add("snapshot_prefix", solver.getProperty("snapshot_prefix"));

        st.add("train_data_itr", getIteratorName("TRAIN"));
        st.add("test_data_itr", getIteratorName("TEST"));

        String context = solver.getProperty("solver_mode", "cpu").toLowerCase();
        context = String.format("mx.%s()", context);
        st.add("ctx", context);

        st.add("loss", loss);

        st.add("data_names", getDataNames());
        st.add("label_names", getLabelNames());

        st.add("init_params", generateInitializer());

        st.add("init_optimizer", generateOptimizer());
        st.add("gamma", solver.getProperty("gamma"));
        st.add("power", solver.getProperty("power"));
        st.add("lr_update", generateLRUpdate());

        return st.render();
    }

    private String generateParamInitializer() {
        return gh.getTemplate("param_initializer").render();
    }

    private String generateMetricsClasses() {
        ST st = gh.getTemplate("metrics_classes");

        String display = solver.getProperty("display");
        String average_loss = solver.getProperty("average_loss");

        if (display != null)
            st.add("display", display);

        if (average_loss != null)
            st.add("average_loss", average_loss);

        return st.render();
    }

    private String generateParamsLoader() {
        return gh.getTemplate("params_loader").render();
    }

    private String getLoss(MLModel model, StringBuilder out) {
        List<String> losses = new ArrayList<>();
        for (Layer layer : model.getLayerList()) {
            if (layer.getType().toLowerCase().endsWith("loss")) {
                losses.add(gh.getVarname(layer.getTop()));
            }
        }

        if (losses.size() == 1) {
            return losses.get(0);
        } else if (losses.size() > 1) {
            String loss_var = "combined_loss";
            ST st = gh.getTemplate("group");
            st.add("var", loss_var);
            st.add("symbols", losses);
            out.append(st.render());
            return loss_var;
        } else {
            System.err.println("No loss found");
            return "unknown_loss";
        }
    }

    private String generateLRUpdate() {
        String code;
        String lrPolicy = solver.getProperty("lr_policy", "fixed").toLowerCase();
        ST st;
        switch (lrPolicy) {
            case "fixed":
                // lr stays fixed. No update needed
                code = "";
                break;
            case "multistep":
                st = gh.getTemplate("lrpolicy_multistep");
                st.add("steps", solver.getProperties("stepvalue"));
                code = st.render();
                break;
            case "step":
            case "exp":
            case "inv":
            case "poly":
            case "sigmoid":
                st = gh.getTemplate("lrpolicy_" + lrPolicy);
                code = st.render();
                break;
            default:
                System.err.println("Unknown lr_policy: " + lrPolicy);
                code = "";
                break;
        }
        return Utils.indent(code, 2, true, 4);
    }

    private String generateValidationMetrics(MLModel mlModel) {
        return new AccuracyMetricsGenerator().generate(mlModel);
    }

    private String generateOptimizer() {
        String caffeOptimizer = solver.getProperty("type", "sgd").toLowerCase();
        ST st;

        String lr = solver.getProperty("base_lr");
        String momentum = solver.getProperty("momentum", "0.9");
        String wd = solver.getProperty("weight_decay", "0.0005");

        switch (caffeOptimizer) {
            case "adadelta":
                st = gh.getTemplate("opt_default");
                st.add("opt_name", "AdaDelta");
                st.add("epsilon", solver.getProperty("delta"));
                break;
            case "adagrad":
                st = gh.getTemplate("opt_default");
                st.add("opt_name", "AdaGrad");
                break;
            case "adam":
                st = gh.getTemplate("opt_default");
                st.add("opt_name", "Adam");
                break;
            case "nesterov":
                st = gh.getTemplate("opt_sgd");
                st.add("opt_name", "NAG");
                st.add("momentum", momentum);
                break;
            case "rmsprop":
                st = gh.getTemplate("opt_default");
                st.add("opt_name", "RMSProp");
                break;
            default:
                if (!caffeOptimizer.equals("sgd"))
                    System.err.println("Unknown optimizer. Will use SGD instead.");

                st = gh.getTemplate("opt_sgd");
                st.add("opt_name", "SGD");
                st.add("momentum", momentum);
                break;
        }
        st.add("lr", lr);
        st.add("wd", wd);

        return st.render();
    }

    private String generateInitializer() {
        ST st = gh.getTemplate("init_params");
        st.add("params_file", paramsFilePath);
        return st.render();
    }

    private String generateImports() {
        return gh.getTemplate("imports").render();
    }

    private StringBuilder generateIterators() {
        StringBuilder code = new StringBuilder();

        for (Layer layer : mlModel.getDataLayers()) {
            String iterator = generateIterator(layer);
            code.append(iterator);
        }

        return code;
    }

    private String getIteratorName(String phase) {
        for (Layer layer : mlModel.getDataLayers()) {
            String layerPhase = layer.getAttr("include.phase", phase);
            if (phase.equalsIgnoreCase(layerPhase)) {
                return layerPhase.toLowerCase() + "_" + layer.getName() + "_" + "itr";
            }
        }
        return null;
    }

    private List<String> getDataNames() {
        return getDataNames(0);
    }

    private List<String> getLabelNames() {
        return getDataNames(1);
    }

    private List<String> getDataNames(int topIndex) {
        List<String> dataList = new ArrayList<String>();
        for (Layer layer : mlModel.getDataLayers()) {
            if (layer.getAttr("include.phase").equalsIgnoreCase("train")) {
                String dataName = layer.getTops().get(topIndex);
                if (dataName != null)
                    dataList.add(String.format("'%s'", dataName));
            }
        }
        return dataList;
    }

    private StringBuilder generateInputVars() {
        StringBuilder code = new StringBuilder();

        Set<String> tops = new HashSet<String>();

        for (Layer layer : mlModel.getDataLayers())
            for (String top : layer.getTops())
                tops.add(top);

        for (String top : tops)
            code.append(gh.generateVar(gh.getVarname(top), top, null, null, null, null));

        code.append(System.lineSeparator());
        return code;
    }

    private String generateIterator(Layer layer) {
        String iteratorName = layer.getAttr("include.phase");
        iteratorName = iteratorName.toLowerCase();
        iteratorName = iteratorName + "_" + layer.getName() + "_" + "itr";

        ST st = stGroup.getInstanceOf("iterator");

        String prototxt = layer.getPrototxt();
        prototxt = prototxt.replace("\r", "");
        prototxt = prototxt.replace("\n", " \\\n");
        prototxt = "'" + prototxt + "'";
        prototxt = Utils.indent(prototxt, 1, true, 4);

        st.add("iter_name", iteratorName);
        st.add("prototxt", prototxt);

        String dataName = "???";
        if (layer.getTops().size() >= 1)
            dataName = layer.getTops().get(0);
        else
            System.err.println(String.format("Data layer %s doesn't have data", layer.getName()));
        st.add("data_name", dataName);

        String labelName = "???";
        if (layer.getTops().size() >= 1)
            labelName = layer.getTops().get(1);
        else
            System.err.println(String.format("Data layer %s doesn't have label", layer.getName()));
        st.add("label_name", labelName);

        if(layer.hasAttr("data_param.num_examples")) {
            st.add("num_examples", layer.getAttr("data_param.num_examples"));
        }

        return st.render();
    }

}
